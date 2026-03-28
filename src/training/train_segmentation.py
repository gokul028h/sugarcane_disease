import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# Ensure module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.segmentation_models import UNet
from src.data.segmentation_loaders import get_segmentation_dataloaders

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs_flat = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def train_segmentation():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 8)
    lr = config.get("learning_rate", 1e-4)
    img_size = config.get("input_size", 256)

    images_dir = config["data_paths"]["seg_images"]
    masks_dir = config["data_paths"]["seg_masks"]

    exp_dir = Path(config["experiments"]["unet_segmentation"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Segmentation Training on {device} [AMP ON] ---")

    train_loader, val_loader = get_segmentation_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        img_size=img_size
    )

    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # OneCycleLR for segmentation convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr*10, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )
    
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=str(exp_dir / "logs"))
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with autocast():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
        
        val_loss /= len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoints_dir / "unet_best.pth")
            print(">>> Saved new best U-Net model")

    writer.close()
    print("--- Segmentation Training Complete ---")

if __name__ == "__main__":
    train_segmentation()

