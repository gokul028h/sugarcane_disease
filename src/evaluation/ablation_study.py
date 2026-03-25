"""
Ablation study framework for IEEE publication.
Tests the contribution of individual components:
1. Baseline (no augmentation, CrossEntropy)
2. + Data augmentation (RandAugment, flips, etc.)  
3. + Focal Loss (vs CrossEntropy)
4. + MixUp/CutMix
5. Full pipeline (all components)

Each experiment trains the same model (EfficientNet-B4) with different settings.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import numpy as np
from pathlib import Path

from src.models.model_factory import get_model
from src.models.losses import FocalLoss
from src.data.data_loaders import get_dataloaders, get_advanced_dataloaders, get_mixup_cutmix
from src.training.trainer import RobustTrainer
from src.evaluation.evaluate_advanced import evaluate_model
from src.evaluation.visualize import plot_ablation_table


def run_ablation_experiment(experiment_name: str, config: dict, device: torch.device,
                            use_advanced_aug: bool = True,
                            use_focal_loss: bool = True,
                            use_mixup_cutmix: bool = True,
                            use_class_weights: bool = True):
    """Run a single ablation experiment."""
    
    BATCH_SIZE = config["batch_size"]
    LR = config["learning_rate"]
    EPOCHS = config["epochs"]
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    ADV_CFG = config["advanced_training"]
    
    # Experiment directory
    EXP_DIR = Path("experiments") / "ablation" / experiment_name
    (EXP_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  ABLATION: {experiment_name}")
    print(f"  Advanced Aug: {use_advanced_aug} | Focal Loss: {use_focal_loss}")
    print(f"  MixUp/CutMix: {use_mixup_cutmix} | Class Weights: {use_class_weights}")
    print(f"{'='*60}")
    
    # Data loaders
    if use_advanced_aug:
        train_loader, val_loader, test_loader, details = get_advanced_dataloaders(
            BATCH_SIZE, IMG_SIZE, use_class_weights=use_class_weights
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, IMG_SIZE)
        details = {"class_weights": None, "classes": None}
    
    # Model (always EfficientNet for fair comparison)
    model = get_model("efficientnet", NUM_CLASSES, pretrained=True)
    
    # Loss
    if use_focal_loss:
        alpha = details["class_weights"].to(device) if details.get("class_weights") is not None else None
        criterion = FocalLoss(alpha=alpha, gamma=ADV_CFG["focal_loss_gamma"])
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # MixUp/CutMix
    mixup_fn = None
    if use_mixup_cutmix:
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
    
    # Evaluate on test set
    # Save checkpoint to the standard location temporarily for evaluate_model
    eval_checkpoint = EXP_DIR / "checkpoints" / "best_model.pth"
    
    # Quick test evaluation
    model.load_state_dict(torch.load(eval_checkpoint, map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader if not use_advanced_aug else \
                                (get_advanced_dataloaders(BATCH_SIZE, IMG_SIZE, use_class_weights=False)[2]):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = correct / total if total > 0 else 0
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    result = {
        "experiment": experiment_name,
        "accuracy": float(test_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "best_val_acc": train_metrics["best_val_accuracy"],
        "training_time": train_metrics.get("total_training_time_seconds", 0)
    }
    
    with open(EXP_DIR / "ablation_result.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return result


def run_full_ablation():
    """Run all ablation experiments."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiments = [
        {
            "name": "baseline_no_aug",
            "use_advanced_aug": False,
            "use_focal_loss": False,
            "use_mixup_cutmix": False,
            "use_class_weights": False,
            "notes": "No augmentation, CrossEntropy loss"
        },
        {
            "name": "with_augmentation",
            "use_advanced_aug": True,
            "use_focal_loss": False,
            "use_mixup_cutmix": False,
            "use_class_weights": False,
            "notes": "+ RandAugment, RandomFlip"
        },
        {
            "name": "with_focal_loss",
            "use_advanced_aug": True,
            "use_focal_loss": True,
            "use_mixup_cutmix": False,
            "use_class_weights": True,
            "notes": "+ Focal Loss with class weights"
        },
        {
            "name": "with_mixup_cutmix",
            "use_advanced_aug": True,
            "use_focal_loss": True,
            "use_mixup_cutmix": True,
            "use_class_weights": True,
            "notes": "+ MixUp/CutMix augmentation"
        },
    ]
    
    results = []
    for exp in experiments:
        result = run_ablation_experiment(
            experiment_name=exp["name"],
            config=config,
            device=DEVICE,
            use_advanced_aug=exp["use_advanced_aug"],
            use_focal_loss=exp["use_focal_loss"],
            use_mixup_cutmix=exp["use_mixup_cutmix"],
            use_class_weights=exp["use_class_weights"]
        )
        result["notes"] = exp.get("notes", "")
        results.append(result)
    
    # Generate ablation table
    results_dir = Path("experiments") / "ieee_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_ablation_table(results, str(results_dir / "ablation_study_table.png"))
    
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print(f"\n{'='*60}")
    print("  ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"\n{'Experiment':<25} {'Accuracy':<12} {'F1':<12} {'Notes'}")
    print("-" * 80)
    for r in results:
        print(f"{r['experiment']:<25} {r['accuracy']:<12.4f} {r['f1']:<12.4f} {r.get('notes', '')}")
    
    return results


if __name__ == "__main__":
    run_full_ablation()
