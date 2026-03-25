"""
Full Pipeline Demo — Single Image End-to-End Inference.
Runs the complete multi-stage pipeline on a single image:
  Input Image → Classification → Grad-CAM → Segmentation → Severity → Output

Usage:
    python run_full_pipeline.py --image path/to/image.jpg
    python run_full_pipeline.py --image path/to/image.jpg --model swin
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import cv2
import yaml
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

from src.models.model_factory import get_model
from src.models.segmentation_models import UNet
from src.explainability.gradcam import generate_gradcam_heatmap
from src.evaluation.severity import calculate_severity


def run_pipeline(image_path: str, model_name: str = "efficientnet", output_dir: str = "experiments/pipeline_demo"):
    """
    Run the full multi-stage pipeline on a single image.
    
    Stage 1: Classification (disease type + probabilities)
    Stage 2: Explainability (Grad-CAM heatmap)
    Stage 3: Segmentation (U-Net diseased region mask)
    Stage 4: Severity estimation (percentage of affected area)
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    
    if model_name == "swin":
        IMG_SIZE = 224
    
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess image
    img_pil = Image.open(image_path).convert("RGB")
    img_display = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.array(img_display, dtype=np.float32) / 255.0
    
    cls_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = cls_transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # Get class names from dataset
    CLASS_NAMES = ["BacterialBlights", "Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
    
    # ─── Stage 1: Classification ─────────────────────────────────
    print("\n[Stage 1] Classification...")
    cls_checkpoint = config["checkpoints"].get(model_name)
    
    cls_model = get_model(model_name, NUM_CLASSES, pretrained=False)
    if cls_checkpoint and os.path.exists(cls_checkpoint):
        cls_model.load_state_dict(torch.load(cls_checkpoint, map_location=DEVICE))
        print(f"  Loaded checkpoint: {cls_checkpoint}")
    else:
        # Try alternate path
        alt_path = Path(config["experiments"][model_name]) / "checkpoints" / "best_model.pth"
        if alt_path.exists():
            cls_model.load_state_dict(torch.load(alt_path, map_location=DEVICE))
            print(f"  Loaded checkpoint: {alt_path}")
        else:
            print(f"  WARNING: No checkpoint found, using untrained model")
    
    cls_model.to(DEVICE)
    cls_model.eval()
    
    with torch.no_grad():
        outputs = cls_model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]
    
    print(f"  Predicted: {pred_class} ({probs[pred_idx]*100:.1f}%)")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {probs[i]*100:.1f}%")
    
    # ─── Stage 2: Grad-CAM Explainability ────────────────────────
    print("\n[Stage 2] Grad-CAM Explainability...")
    gradcam_heatmap = None
    gradcam_overlay = None
    try:
        gradcam_heatmap, gradcam_overlay = generate_gradcam_heatmap(
            cls_model, model_name, input_tensor, img_arr, target_class=pred_idx
        )
        print(f"  Heatmap generated successfully")
    except Exception as e:
        print(f"  Grad-CAM failed: {e}")
    
    # ─── Stage 3: Segmentation ───────────────────────────────────
    print("\n[Stage 3] Segmentation...")
    seg_mask = None
    binary_mask = None
    
    seg_checkpoint = Path(config["experiments"]["unet_segmentation"]) / "checkpoints" / "unet_best.pth"
    if seg_checkpoint.exists() and pred_class != "Healthy":
        seg_model = UNet(n_channels=3, n_classes=1).to(DEVICE)
        seg_model.load_state_dict(torch.load(seg_checkpoint, map_location=DEVICE))
        seg_model.eval()
        
        # Use same input tensor (already normalized)
        seg_input = transforms.Compose([
            transforms.Resize((config["input_size"], config["input_size"])),
            transforms.ToTensor(),
        ])(img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            seg_out = seg_model(seg_input)
            seg_mask = torch.sigmoid(seg_out).cpu().numpy()[0, 0]
            binary_mask = (seg_mask > 0.5).astype(np.uint8)
        
        print(f"  Segmentation mask generated")
    elif pred_class == "Healthy":
        print(f"  Skipped (leaf is healthy)")
        binary_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    else:
        print(f"  No segmentation checkpoint found at {seg_checkpoint}")
    
    # ─── Stage 4: Severity Estimation ────────────────────────────
    print("\n[Stage 4] Severity Estimation...")
    severity = 0.0
    if binary_mask is not None:
        severity = calculate_severity(binary_mask)
    print(f"  Disease Severity: {severity:.2f}%")
    
    # ─── Generate Visualization ──────────────────────────────────
    print("\n[Output] Generating visualization...")
    
    num_panels = 5
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4.5))
    
    # Panel 1: Original
    axes[0].imshow(img_arr)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Classification result
    axes[1].imshow(img_arr)
    axes[1].set_title(f'Class: {pred_class}\n({probs[pred_idx]*100:.1f}%)', 
                       fontweight='bold', color='darkgreen')
    axes[1].axis('off')
    
    # Panel 3: Grad-CAM
    if gradcam_overlay is not None:
        axes[2].imshow(gradcam_overlay)
    else:
        axes[2].imshow(img_arr)
        axes[2].text(0.5, 0.5, 'N/A', transform=axes[2].transAxes,
                     ha='center', va='center', fontsize=20, color='red')
    axes[2].set_title('Grad-CAM Heatmap', fontweight='bold')
    axes[2].axis('off')
    
    # Panel 4: Segmentation
    if seg_mask is not None:
        axes[3].imshow(seg_mask, cmap='hot')
    else:
        axes[3].imshow(np.zeros_like(img_arr[:,:,0]), cmap='gray')
        axes[3].text(0.5, 0.5, 'N/A', transform=axes[3].transAxes,
                     ha='center', va='center', fontsize=20, color='red')
    axes[3].set_title('Segmentation Mask', fontweight='bold')
    axes[3].axis('off')
    
    # Panel 5: Severity overlay
    if binary_mask is not None and seg_mask is not None:
        overlay = img_arr.copy() if img_arr.shape[:2] == binary_mask.shape else \
                  cv2.resize(img_arr, (binary_mask.shape[1], binary_mask.shape[0]))
        mask_resized = binary_mask.astype(np.float32)
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_resized * 0.4, 0, 1)
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] - mask_resized * 0.2, 0, 1)
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] - mask_resized * 0.2, 0, 1)
        axes[4].imshow(overlay)
    else:
        axes[4].imshow(img_arr)
    axes[4].set_title(f'Severity: {severity:.1f}%', fontweight='bold', 
                       color='red' if severity > 20 else 'orange' if severity > 5 else 'green')
    axes[4].axis('off')
    
    plt.suptitle('Multi-Stage Sugarcane Disease Detection Pipeline', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"pipeline_result_{Path(image_path).stem}.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # Save JSON result
    result = {
        "image_path": str(image_path),
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "severity_percentage": severity,
        "visualization_path": str(output_path)
    }
    
    json_path = OUTPUT_DIR / f"pipeline_result_{Path(image_path).stem}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Disease: {pred_class} ({probs[pred_idx]*100:.1f}%)")
    print(f"  Severity: {severity:.2f}%")
    print(f"  Visualization: {output_path}")
    print(f"  JSON: {json_path}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full multi-stage pipeline on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="efficientnet", help="Classification model to use")
    parser.add_argument("--output", type=str, default="experiments/pipeline_demo", help="Output directory")
    args = parser.parse_args()
    
    run_pipeline(args.image, args.model, args.output)
