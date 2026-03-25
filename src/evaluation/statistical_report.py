"""
Statistical validation report using K-Fold cross-validation results.
Aggregates per-fold metrics and reports mean ± std for IEEE publication.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.optim as optim
import numpy as np
import yaml
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.model_factory import get_model
from src.models.losses import FocalLoss
from src.data.data_loaders import get_advanced_dataloaders, get_mixup_cutmix
from src.training.trainer import RobustTrainer
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def run_statistical_cv(model_name: str = "efficientnet"):
    """
    Run K-Fold cross-validation with comprehensive per-fold metric collection.
    
    Reports mean ± std for: Accuracy, Precision, Recall, F1-Score
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    BATCH_SIZE = config["batch_size"]
    LR = config["learning_rate"]
    EPOCHS = config["epochs"]
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    ADV_CFG = config["advanced_training"]
    K_FOLDS = ADV_CFG["k_folds"]
    
    if model_name == "swin":
        IMG_SIZE = 224
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load full training dataset
    train_loader, _, _, details = get_advanced_dataloaders(BATCH_SIZE, IMG_SIZE, use_class_weights=False)
    dataset = train_loader.dataset
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    targets = dataset.targets
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1}/{K_FOLDS} — {model_name.upper()}")
        print(f"{'='*60}")
        
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        
        train_dl = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        model = get_model(model_name, NUM_CLASSES, pretrained=True)
        
        alpha = details["class_weights"].to(DEVICE) if details.get("class_weights") is not None else None
        criterion = FocalLoss(alpha=alpha, gamma=ADV_CFG["focal_loss_gamma"])
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        mixup_fn = get_mixup_cutmix(NUM_CLASSES, ADV_CFG["mixup_alpha"], ADV_CFG["cutmix_alpha"])
        
        fold_dir = Path(config["experiments"][model_name]) / f"cv_fold_{fold+1}"
        (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        
        trainer = RobustTrainer(
            model=model,
            device=DEVICE,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            patience=ADV_CFG["early_stopping_patience"],
            exp_dir=str(fold_dir),
            mixup_cutmix_fn=mixup_fn
        )
        
        train_metrics = trainer.fit(train_dl, val_dl, EPOCHS)
        
        # Evaluate on validation fold with detailed metrics
        best_checkpoint = fold_dir / "checkpoints" / "best_model.pth"
        model.load_state_dict(torch.load(best_checkpoint, map_location=DEVICE))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        
        fold_result = {
            "fold": fold + 1,
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "best_val_loss": train_metrics["best_val_loss"],
            "training_time_seconds": train_metrics.get("total_training_time_seconds", 0)
        }
        fold_results.append(fold_result)
        
        print(f"  Fold {fold+1} — Acc: {acc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
    
    # Aggregate results
    accs = [r["accuracy"] for r in fold_results]
    precs = [r["precision"] for r in fold_results]
    recs = [r["recall"] for r in fold_results]
    f1s = [r["f1_score"] for r in fold_results]
    
    summary = {
        "model": model_name,
        "k_folds": K_FOLDS,
        "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
        "precision": {"mean": float(np.mean(precs)), "std": float(np.std(precs))},
        "recall": {"mean": float(np.mean(recs)), "std": float(np.std(recs))},
        "f1_score": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
        "per_fold": fold_results
    }
    
    # Save
    results_dir = Path("experiments") / "ieee_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"cv_results_{model_name}.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    # Generate visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    folds = [r["fold"] for r in fold_results]
    
    # Bar chart of per-fold accuracy
    bars = axes[0].bar(folds, accs, color='#4472C4', edgecolor='white', alpha=0.85)
    axes[0].axhline(y=np.mean(accs), color='red', linestyle='--', linewidth=2, 
                     label=f'Mean: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{K_FOLDS}-Fold CV Accuracy — {model_name}')
    axes[0].set_ylim([0, 1.05])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Grouped bar chart of all metrics
    x = np.arange(K_FOLDS)
    width = 0.2
    axes[1].bar(x - 1.5*width, accs, width, label='Accuracy', color='#4472C4')
    axes[1].bar(x - 0.5*width, precs, width, label='Precision', color='#ED7D31')
    axes[1].bar(x + 0.5*width, recs, width, label='Recall', color='#A5A5A5')
    axes[1].bar(x + 1.5*width, f1s, width, label='F1-Score', color='#70AD47')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'Per-Fold Metrics — {model_name}')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Fold {i+1}' for i in range(K_FOLDS)])
    axes[1].legend(fontsize=9)
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(results_dir / f"cv_results_{model_name}.png", dpi=300)
    plt.close(fig)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  {K_FOLDS}-FOLD CROSS-VALIDATION RESULTS — {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  F1-Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"{'='*60}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet")
    args = parser.parse_args()
    run_statistical_cv(args.model)
