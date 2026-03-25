"""
Failure case analysis module.
Identifies misclassified samples from the test set and generates visualizations
explaining possible reasons for failure (with Grad-CAM overlays).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import yaml
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.model_factory import get_model
from src.explainability.gradcam import generate_gradcam_heatmap


def analyze_failures(model_name: str = "efficientnet", max_failures: int = 20):
    """
    Find and visualize misclassified test images with Grad-CAM explanations.
    
    Args:
        model_name: Model to analyze
        max_failures: Maximum number of failure cases to visualize
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    if model_name == "swin":
        IMG_SIZE = 224
    
    EXP_DIR = Path(config["experiments"][model_name])
    CHECKPOINT = EXP_DIR / "checkpoints" / "best_model.pth"
    FAILURE_DIR = EXP_DIR / "failure_analysis"
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CHECKPOINT.exists():
        print(f"No checkpoint found at {CHECKPOINT}")
        return
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    display_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])
    
    # Load dataset with paths
    test_dataset = datasets.ImageFolder(root="data/test", transform=eval_transform)
    class_names = test_dataset.classes
    
    # Load model
    model = get_model(model_name, NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    # Find misclassifications
    failures = []
    correct_count = 0
    total_count = 0
    
    print(f"Analyzing {model_name} predictions on test set...")
    for idx in tqdm(range(len(test_dataset)), desc="Scanning test set"):
        img_tensor, true_label = test_dataset[idx]
        img_path = test_dataset.samples[idx][0]
        
        with torch.no_grad():
            input_t = img_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_t)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_label = int(probs.argmax())
        
        total_count += 1
        if pred_label == true_label:
            correct_count += 1
        else:
            failures.append({
                "index": idx,
                "path": img_path,
                "true_label": true_label,
                "true_class": class_names[true_label],
                "pred_label": pred_label,
                "pred_class": class_names[pred_label],
                "confidence": float(probs[pred_label]),
                "true_class_prob": float(probs[true_label]),
                "all_probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            })
    
    print(f"\nTotal: {total_count}, Correct: {correct_count}, Failures: {len(failures)}")
    print(f"Accuracy: {correct_count/total_count:.4f}")
    
    # Sort by confidence (most confident mistakes first — most interesting)
    failures.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Visualize top failures with Grad-CAM
    num_to_show = min(max_failures, len(failures))
    
    if num_to_show > 0:
        # Create a grid: 4 columns per row
        cols = 4
        rows = (num_to_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.5 * rows))
        if rows == 1:
            axes = axes[np.newaxis, :] if cols > 1 else np.array([[axes]])
        
        for i in range(num_to_show):
            row, col = divmod(i, cols)
            ax = axes[row, col]
            
            failure = failures[i]
            
            # Load original image for display
            img_pil = Image.open(failure["path"]).convert("RGB")
            img_display = display_transform(img_pil)
            img_arr = np.array(img_display, dtype=np.float32) / 255.0
            
            # Generate Grad-CAM
            img_tensor = eval_transform(img_pil).unsqueeze(0).to(DEVICE)
            try:
                _, gradcam_vis = generate_gradcam_heatmap(
                    model, model_name, img_tensor, img_arr, 
                    target_class=failure["pred_label"]
                )
                # gradcam_vis is uint8 RGB
                ax.imshow(gradcam_vis)
            except Exception:
                ax.imshow(img_arr)
            
            ax.set_title(
                f"True: {failure['true_class']}\n"
                f"Pred: {failure['pred_class']} ({failure['confidence']:.2f})",
                fontsize=9, color='red', fontweight='bold'
            )
            ax.axis('off')
        
        # Hide unused axes
        for i in range(num_to_show, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].axis('off')
        
        plt.suptitle(f'Failure Case Analysis — {model_name}\n(Grad-CAM highlighting predicted class)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(FAILURE_DIR / "failure_cases.png", dpi=200)
        plt.close(fig)
    
    # Save failure details as JSON
    save_failures = [{
        "path": f["path"],
        "true_class": f["true_class"],
        "pred_class": f["pred_class"],
        "confidence": f["confidence"],
        "true_class_prob": f["true_class_prob"]
    } for f in failures[:max_failures]]
    
    with open(FAILURE_DIR / "failure_details.json", "w") as f:
        json.dump(save_failures, f, indent=4)
    
    # Confusion breakdown
    confusion_pairs = {}
    for f in failures:
        pair = f"{f['true_class']} → {f['pred_class']}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nMost Common Misclassification Pairs:")
    print("-" * 50)
    for pair, count in sorted_pairs[:10]:
        print(f"  {pair}: {count}")
    
    with open(FAILURE_DIR / "confusion_pairs.json", "w") as f:
        json.dump(dict(sorted_pairs), f, indent=4)
    
    print(f"\nFailure analysis saved to {FAILURE_DIR}")
    return failures


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet")
    parser.add_argument("--max-failures", type=int, default=20)
    args = parser.parse_args()
    analyze_failures(args.model, args.max_failures)
