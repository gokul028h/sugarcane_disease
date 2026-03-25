"""
Advanced model evaluation with comprehensive metrics and IEEE-quality visualizations.
Generates: Classification report, Confusion matrix, ROC curves, PR curves,
training curves, and exports all metrics as JSON.
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

from src.models.model_factory import get_model
from src.data.data_loaders import get_advanced_dataloaders
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.visualize import (
    plot_training_curves, plot_confusion_matrix,
    plot_roc_curves, plot_pr_curves
)


def evaluate_model(model_name: str, checkpoint_path: str = None):
    """
    Comprehensive evaluation of a trained classification model.
    
    Args:
        model_name: Model identifier (efficientnet, swin, convnext, etc.)
        checkpoint_path: Override checkpoint path. If None, uses default.
    
    Returns:
        dict: Complete evaluation metrics
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    BATCH_SIZE = config["batch_size"]
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    
    # Override image size for Swin
    if model_name == "swin":
        IMG_SIZE = 224
    
    EXP_DIR = Path(config["experiments"][model_name])
    
    if checkpoint_path is None:
        checkpoint_path = EXP_DIR / "checkpoints" / "best_model.pth"
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found for {model_name} at {checkpoint_path}")
        return None

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    _, _, test_loader, details = get_advanced_dataloaders(BATCH_SIZE, IMG_SIZE, use_class_weights=False)
    classes = details["classes"]

    # Load model
    model = get_model(model_name, NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print(f"Evaluating {model_name} on Test Set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute all metrics
    metrics = compute_classification_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_probs=np.array(all_probs),
        class_names=classes
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1-Score:   {metrics['macro_avg']['f1_score']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_avg']['f1_score']:.4f}")
    print(f"{'='*60}")
    print(metrics['classification_report_str'])
    
    # --- Generate Visualizations ---
    vis_dir = EXP_DIR / "figures"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, classes, str(vis_dir / "confusion_matrix.png"), model_name)
    
    # ROC Curves
    plot_roc_curves(metrics['roc_data'], str(vis_dir / "roc_curves.png"), model_name)
    
    # PR Curves
    plot_pr_curves(metrics['pr_data'], str(vis_dir / "pr_curves.png"), model_name)
    
    # Training curves (if history exists)
    metrics_file = EXP_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            train_metrics = json.load(f)
        if "history" in train_metrics:
            plot_training_curves(train_metrics["history"], str(vis_dir), model_name)
    
    # Save classification report
    with open(EXP_DIR / "classification_report.txt", "w") as f:
        f.write(metrics['classification_report_str'])
    
    # Save full metrics JSON (exclude large arrays for readability)
    save_metrics = {
        "model_name": model_name,
        "accuracy": metrics['accuracy'],
        "per_class": metrics['per_class'],
        "macro_avg": metrics['macro_avg'],
        "weighted_avg": metrics['weighted_avg'],
        "confusion_matrix": metrics['confusion_matrix'],
        "roc_auc_per_class": {k: v['auc'] for k, v in metrics['roc_data'].items()},
        "average_precision_per_class": {k: v['average_precision'] for k, v in metrics['pr_data'].items()}
    }
    with open(EXP_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=4)
    
    print(f"\nAll evaluation artifacts saved to {EXP_DIR}")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained classification model")
    parser.add_argument("--model", type=str, default="efficientnet", 
                        help="Model name to evaluate (efficientnet, swin, convnext, baseline_cnn, etc.)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path")
    args = parser.parse_args()
    evaluate_model(args.model, args.checkpoint)
