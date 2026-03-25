"""
Multi-model comparison framework.
Trains and evaluates all supported architectures and generates a unified comparison table.

Supported models: baseline_cnn, resnet_frozen, efficientnet, convnext, swin
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.optim as optim
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.models.model_factory import get_model
from src.models.losses import FocalLoss
from src.data.data_loaders import get_advanced_dataloaders, get_mixup_cutmix
from src.training.trainer import RobustTrainer
from src.evaluation.evaluate_advanced import evaluate_model
from src.evaluation.visualize import plot_model_comparison_table, plot_multi_model_curves


MODELS_TO_COMPARE = ["efficientnet", "convnext", "swin"]


def train_single_model(model_name: str, config: dict, device: torch.device):
    """Train a single model and return metrics."""
    
    BATCH_SIZE = config["batch_size"]
    LR = config["learning_rate"]
    EPOCHS = config["epochs"]
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    ADV_CFG = config["advanced_training"]
    
    # Override for Swin
    if model_name == "swin":
        IMG_SIZE = 224
    
    EXP_DIR = Path(config["experiments"][model_name])
    (EXP_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Data
    train_loader, val_loader, _, details = get_advanced_dataloaders(
        BATCH_SIZE, IMG_SIZE, use_class_weights=True
    )
    
    # Model
    model = get_model(model_name, NUM_CLASSES, pretrained=True)
    
    # Loss
    alpha = details["class_weights"].to(device) if details["class_weights"] is not None else None
    criterion = FocalLoss(alpha=alpha, gamma=ADV_CFG["focal_loss_gamma"])
    
    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Augmentation
    mixup_fn = get_mixup_cutmix(NUM_CLASSES, ADV_CFG["mixup_alpha"], ADV_CFG["cutmix_alpha"])
    
    # Train
    trainer = RobustTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=ADV_CFG["early_stopping_patience"],
        exp_dir=str(EXP_DIR),
        mixup_cutmix_fn=mixup_fn
    )
    
    train_metrics = trainer.fit(train_loader, val_loader, EPOCHS)
    return train_metrics


def run_model_comparison(models: list = None, skip_training: bool = False):
    """
    Run training + evaluation for all specified models and generate comparison.
    
    Args:
        models: List of model names. Defaults to MODELS_TO_COMPARE.
        skip_training: If True, only evaluate (assumes checkpoints exist).
    """
    if models is None:
        models = MODELS_TO_COMPARE
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    comparison_results = []
    all_histories = {}
    
    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*60}")
        
        # Training phase
        if not skip_training:
            train_metrics = train_single_model(model_name, config, DEVICE)
            if "history" in train_metrics:
                all_histories[model_name] = train_metrics["history"]
        else:
            # Load existing metrics
            exp_dir = Path(config["experiments"][model_name])
            metrics_file = exp_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    train_metrics = json.load(f)
                if "history" in train_metrics:
                    all_histories[model_name] = train_metrics["history"]
            else:
                train_metrics = {}
        
        # Evaluation phase
        eval_metrics = evaluate_model(model_name)
        
        if eval_metrics is not None:
            result = {
                "model": model_name,
                "accuracy": eval_metrics["accuracy"],
                "precision": eval_metrics["macro_avg"]["precision"],
                "recall": eval_metrics["macro_avg"]["recall"],
                "f1": eval_metrics["macro_avg"]["f1_score"],
                "params_M": train_metrics.get("total_parameters", 0) / 1e6,
                "training_time_min": train_metrics.get("total_training_time_seconds", 0) / 60.0,
                "per_class": eval_metrics["per_class"],
                "roc_auc": {k: v["auc"] for k, v in eval_metrics.get("roc_data", {}).items()},
            }
            comparison_results.append(result)
    
    # Generate comparison outputs
    results_dir = Path("experiments") / "ieee_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if comparison_results:
        # Table figure
        plot_model_comparison_table(comparison_results, str(results_dir / "model_comparison_table.png"))
        
        # Multi-model curves
        if all_histories:
            plot_multi_model_curves(all_histories, str(results_dir))
        
        # Save raw data
        # Convert non-serializable types
        save_results = []
        for r in comparison_results:
            save_r = {k: v for k, v in r.items()}
            save_results.append(save_r)
        
        with open(results_dir / "model_comparison.json", "w") as f:
            json.dump(save_results, f, indent=4, default=str)
        
        print(f"\n{'='*60}")
        print("  MODEL COMPARISON COMPLETE")
        print(f"{'='*60}")
        print(f"\nResults saved to {results_dir}")
        
        # Print summary table
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Params(M)':<12}")
        print("-" * 80)
        for r in comparison_results:
            print(f"{r['model']:<20} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
                  f"{r['recall']:<12.4f} {r['f1']:<12.4f} {r['params_M']:<12.2f}")
    
    return comparison_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run model comparison experiments")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to compare (default: efficientnet convnext swin)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    args = parser.parse_args()
    
    run_model_comparison(models=args.models, skip_training=args.skip_training)
