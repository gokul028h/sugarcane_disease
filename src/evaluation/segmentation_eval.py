"""
Segmentation evaluation module.
Evaluates U-Net segmentation on pseudo-mask dataset using IoU and Dice metrics.
Generates qualitative visualization grids for IEEE publication.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import yaml
import json
from pathlib import Path
from tqdm import tqdm

from src.models.segmentation_models import UNet
from src.data.segmentation_loaders import get_segmentation_dataloaders
from src.evaluation.metrics import compute_segmentation_metrics, compute_iou, compute_dice
from src.evaluation.visualize import plot_segmentation_grid


def evaluate_segmentation():
    """Evaluate the trained U-Net segmentation model."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = config["input_size"]
    
    images_dir = config["data_paths"]["seg_images"]
    masks_dir = config["data_paths"]["seg_masks"]
    
    EXP_DIR = Path(config["experiments"]["unet_segmentation"])
    CHECKPOINT = EXP_DIR / "checkpoints" / "unet_best.pth"
    
    if not CHECKPOINT.exists():
        print(f"No segmentation checkpoint found at {CHECKPOINT}")
        return None
    
    # Load model
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    # Load data
    _, val_loader = get_segmentation_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=8,
        img_size=IMG_SIZE
    )
    
    all_pred_masks = []
    all_gt_masks = []
    sample_images = []
    sample_pred_masks = []
    sample_gt_masks = []
    
    print("Evaluating segmentation model...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Segmentation Eval")):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            pred_probs = torch.sigmoid(outputs).cpu().numpy()
            gt_np = masks.cpu().numpy()
            
            for i in range(pred_probs.shape[0]):
                pred_mask = pred_probs[i, 0]  # (H, W)
                gt_mask = gt_np[i, 0]  # (H, W)
                
                all_pred_masks.append(pred_mask)
                all_gt_masks.append(gt_mask)
                
                # Collect samples for visualization
                if len(sample_images) < 8:
                    img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize if needed
                    img_np = np.clip(img_np, 0, 1)
                    sample_images.append(img_np)
                    sample_pred_masks.append(pred_mask)
                    sample_gt_masks.append(gt_mask)
    
    # Compute metrics
    seg_metrics = compute_segmentation_metrics(all_pred_masks, all_gt_masks, threshold=0.5)
    
    print(f"\n--- Segmentation Results ---")
    print(f"Mean IoU:  {seg_metrics['mean_iou']:.4f} ± {seg_metrics['std_iou']:.4f}")
    print(f"Mean Dice: {seg_metrics['mean_dice']:.4f} ± {seg_metrics['std_dice']:.4f}")
    
    # Save metrics
    save_metrics = {
        "mean_iou": seg_metrics['mean_iou'],
        "std_iou": seg_metrics['std_iou'],
        "mean_dice": seg_metrics['mean_dice'],
        "std_dice": seg_metrics['std_dice']
    }
    with open(EXP_DIR / "segmentation_metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=4)
    
    # Generate visualization grid
    if sample_images:
        # Create overlays
        overlays = []
        for img, pred in zip(sample_images, sample_pred_masks):
            binary = (pred > 0.5).astype(np.float32)
            overlay = img.copy()
            # Red channel overlay for diseased regions
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + binary * 0.4, 0, 1)
            overlay[:, :, 1] = np.clip(overlay[:, :, 1] - binary * 0.2, 0, 1)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2] - binary * 0.2, 0, 1)
            overlays.append(overlay)
        
        # Use GT masks as "Grad-CAM" placeholder and pseudo masks
        plot_segmentation_grid(
            images=sample_images,
            gradcams=sample_gt_masks,  # Using GT masks as reference
            pseudo_masks=sample_gt_masks,
            pred_masks=sample_pred_masks,
            overlays=overlays,
            save_path=str(EXP_DIR / "segmentation_visualization.png"),
            num_samples=min(4, len(sample_images))
        )
        print(f"Visualization saved to {EXP_DIR / 'segmentation_visualization.png'}")
    
    return seg_metrics


if __name__ == "__main__":
    evaluate_segmentation()
