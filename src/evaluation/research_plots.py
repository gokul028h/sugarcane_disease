import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import json
from pathlib import Path
import pandas as pd

class ResearchPlotter:
    """
    Generates IEEE publication-grade visualizations and tables.
    """
    def __init__(self, exp_dir: str):
        self.exp_dir = Path(exp_dir)
        self.plot_dir = self.exp_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_history(self, history: dict):
        """Plots training vs validation loss and accuracy curves."""
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / "learning_curves.png", dpi=300)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Generates a labeled heatmap for the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(self.plot_dir / "confusion_matrix.png", dpi=300)
        plt.close()

    def plot_roc_curves(self, y_true, y_probs, class_names):
        """Plots multi-class ROC curves."""
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            # One-vs-Rest ROC
            y_true_binary = (np.array(y_true) == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.savefig(self.plot_dir / "roc_curves.png", dpi=300)
        plt.close()

def generate_comparison_table(experiments_root: str, output_path: str):
    """
    Parses all experiment folders and generates a summary comparison table.
    """
    results = []
    root = Path(experiments_root)
    
    for exp_folder in root.iterdir():
        if exp_folder.is_dir() and (exp_folder / "metrics.json").exists():
            with open(exp_folder / "metrics.json", "r") as f:
                data = json.load(f)
                results.append({
                    "Model": exp_folder.name,
                    "Accuracy": f"{data.get('best_val_accuracy', 0)*100:.2f}%",
                    "Params (M)": f"{data.get('total_parameters', 0)/1e6:.2f}",
                    "Train Time (s)": data.get('total_training_time_seconds', 0),
                    "Loss": f"{data.get('best_val_loss', 0):.4f}"
                })
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Comparison table saved to {output_path}")
        return df
    return None

def visualize_multi_stage_results(image, heatmap, mask, save_path):
    """
    Creates a publication-style visualization grid for a single prediction.
    Cols: Original | Grad-CAM | Segmentation Mask | Overlay
    """
    image = np.array(image) / 255.0 if np.max(image) > 1 else image
    
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM Activation")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Diseased Mask")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    # Overlay logic
    overlay = image.copy()
    overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5 # Red tint
    plt.imshow(overlay)
    plt.title("Severity Visualization")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
