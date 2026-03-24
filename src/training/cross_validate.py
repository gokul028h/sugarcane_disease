import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.optim as optim
import yaml
from pathlib import Path
from src.models.model_factory import get_model
from src.models.losses import FocalLoss
from src.data.data_loaders import get_advanced_dataloaders, get_mixup_cutmix
from src.training.trainer import RobustTrainer
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np

def run_cross_validation(model_name: str):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    BATCH_SIZE = config["batch_size"]
    LR = config["learning_rate"]
    EPOCHS = config["epochs"]
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    ADV_CFG = config["advanced_training"]
    K_FOLDS = ADV_CFG["k_folds"]
    
    EXP_DIR = Path(config["experiments"][model_name])
    (EXP_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We only need the train dataset for cross-val
    train_loader, _, _, details = get_advanced_dataloaders(BATCH_SIZE, IMG_SIZE, use_class_weights=False)
    dataset = train_loader.dataset
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    targets = dataset.targets
    
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n--- Fold {fold + 1}/{K_FOLDS} ---")
        
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        
        # Sampler could be added to train_sub loader if class imbalance handling is needed inside folds
        train_dl = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = get_model(model_name, NUM_CLASSES, pretrained=True)
        criterion = FocalLoss(alpha=details["class_weights"].to(DEVICE), gamma=ADV_CFG["focal_loss_gamma"])
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        mixup_fn = get_mixup_cutmix(NUM_CLASSES, ADV_CFG["mixup_alpha"], ADV_CFG["cutmix_alpha"])

        fold_exp_dir = EXP_DIR / f"fold_{fold+1}"
        (fold_exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        trainer = RobustTrainer(
            model=model,
            device=DEVICE,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            patience=ADV_CFG["early_stopping_patience"],
            exp_dir=str(fold_exp_dir),
            mixup_cutmix_fn=mixup_fn
        )

        metrics = trainer.fit(train_dl, val_dl, EPOCHS)
        fold_metrics.append(metrics["best_val_accuracy"])

    avg_acc = np.mean(fold_metrics)
    print(f"\n--- Cross Validation Complete ---")
    print(f"Average CV Accuracy: {avg_acc:.4f} +/- {np.std(fold_metrics):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet", help="Model name to cross-validate")
    args = parser.parse_args()
    
    run_cross_validation(args.model)
