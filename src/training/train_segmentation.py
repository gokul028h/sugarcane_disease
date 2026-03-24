import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from tqdm import tqdm
from src.models.segmentation_models import UNet
from src.data.segmentation_loaders import get_segmentation_dataloaders
from torch.utils.tensorboard import SummaryWriter

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs are logits. passed through sigmoid first
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

import torch.nn.functional as F

def train_segmentation():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    epochs = 20
    batch_size = 8
    lr = 1e-4
    img_size = config["input_size"]

    images_dir = config["data_paths"]["seg_images"]
    masks_dir = config["data_paths"]["seg_masks"]

    exp_dir = Path(config["experiments"]["unet_segmentation"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading segmentation datasets...")
    train_loader, val_loader = get_segmentation_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        img_size=img_size
    )

    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    writer = SummaryWriter(log_dir=str(exp_dir / "logs"))
    best_val_loss = float('inf')

    print(f"Starting Training on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for images, masks in pbar_val:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoints_dir / "unet_best.pth")
            print(">>> Saved new best U-Net model")

    writer.close()
    print("Training Complete!")

if __name__ == "__main__":
    train_segmentation()
