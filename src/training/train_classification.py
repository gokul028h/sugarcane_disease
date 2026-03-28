import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import numpy as np

# Ensure module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.model_factory import get_model
from src.models.losses import FocalLoss
from src.data.data_loaders import get_advanced_dataloaders, get_mixup_cutmix
from src.training.trainer import RobustTrainer
from src.evaluation.research_plots import ResearchPlotter

def train_model(model_name: str):
    # 1. Load Configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Training Hyperparameters
    BATCH_SIZE = config.get("batch_size", 16)
    LR = config.get("learning_rate", 1e-4)
    EPOCHS = config.get("epochs", 50)
    NUM_CLASSES = config.get("num_classes", 6)
    IMG_SIZE = config.get("input_size", 256)
    ADV_CFG = config.get("advanced_training", {})
    
    # Correct input size for Swin
    if model_name == 'swin':
        IMG_SIZE = 224

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXP_DIR = Path(config["experiments"].get(model_name, f"experiments/{model_name}"))
    
    print(f"\n--- Starting Training: {model_name} on {DEVICE} ---")
    print(f"Metrics will be saved to: {EXP_DIR}")

    # 2. Data Loaders with Pin Memory and Workers for max GPU capacity
    train_loader, val_loader, test_loader, details = get_advanced_dataloaders(
        batch_size=BATCH_SIZE, 
        img_size=IMG_SIZE, 
        use_class_weights=True
    )
    
    # 3. Model, Loss, and Optimizer
    model = get_model(model_name, NUM_CLASSES, pretrained=True)
    
    # Focal Loss for imbalanced disease detection
    class_weights = details["class_weights"].to(DEVICE) if details["class_weights"] is not None else None
    criterion = FocalLoss(alpha=class_weights, gamma=ADV_CFG.get("focal_loss_gamma", 2.0))
    
    # MixUp/CutMix for regularization
    mixup_cutmix_fn = get_mixup_cutmix(
        num_classes=NUM_CLASSES, 
        mixup_alpha=ADV_CFG.get("mixup_alpha", 0.2), 
        cutmix_alpha=ADV_CFG.get("cutmix_alpha", 1.0)
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # OneCycleLR for high-performance convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR*10, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS
    )

    # 4. Robust Training Loop with AMP (Automatic Mixed Precision)
    trainer = RobustTrainer(
        model=model,
        device=DEVICE,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=ADV_CFG.get("early_stopping_patience", 5),
        exp_dir=str(EXP_DIR),
        mixup_cutmix_fn=mixup_cutmix_fn
    )

    metrics = trainer.fit(train_loader, val_loader, EPOCHS)

    # 5. Scientific Visualization
    plotter = ResearchPlotter(str(EXP_DIR))
    plotter.plot_history(metrics["history"])
    
    # Generate Confusion Matrix and ROC on Test Set
    model.load_state_dict(torch.load(EXP_DIR / "checkpoints" / "best_model.pth")['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    plotter.plot_confusion_matrix(all_labels, all_preds, details["classes"])
    plotter.plot_roc_curves(all_labels, np.array(all_probs), details["classes"])
    
    print(f"--- Training Complete for {model_name} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="swin", help="Model name as per config (swin, resnet_finetuned, etc.)")
    args = parser.parse_args()
    
    train_model(args.model)
