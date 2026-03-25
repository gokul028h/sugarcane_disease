"""
IEEE publication-quality visualization module.
Generates all plots needed for the research paper:
- Training/Validation loss and accuracy curves
- ROC curves
- Precision-Recall curves
- Confusion matrices (heatmap)
- Segmentation visualization grids
- Model comparison charts
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json

# IEEE publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def plot_training_curves(history: dict, save_dir: str, model_name: str = ""):
    """
    Plot training vs validation loss and accuracy curves.
    
    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_dir: directory to save plots
        model_name: model identifier for title
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-s', markersize=4, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training and Validation Loss — {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir / "loss_curves.png")
    plt.close(fig)
    
    # Accuracy curves
    fig, ax = plt.subplots(figsize=(8, 5))
    if any(v > 0 for v in history.get('train_acc', [])):
        ax.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Training Accuracy', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-s', markersize=4, label='Validation Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Training and Validation Accuracy — {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    fig.savefig(save_dir / "accuracy_curves.png")
    plt.close(fig)
    
    # Learning rate schedule
    if 'lr' in history and history['lr']:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history['lr'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'Learning Rate Schedule — {model_name}')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        fig.savefig(save_dir / "lr_schedule.png")
        plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path, model_name=""):
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize for display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot with both raw counts and normalized percentages
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.savefig(save_path)
    plt.close(fig)
    
    # Also save normalized version
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2,
                linewidths=0.5, linecolor='white', vmin=0, vmax=1)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title(f'Normalized Confusion Matrix — {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    norm_path = str(save_path).replace('.png', '_normalized.png')
    fig2.savefig(norm_path)
    plt.close(fig2)


def plot_roc_curves(roc_data, save_path, model_name=""):
    """Plot ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    
    for (class_name, data), color in zip(roc_data.items(), colors):
        ax.plot(data['fpr'], data['tpr'], color=color, lw=2,
                label=f'{class_name} (AUC = {data["auc"]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves (One-vs-Rest) — {model_name}')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)


def plot_pr_curves(pr_data, save_path, model_name=""):
    """Plot Precision-Recall curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(pr_data)))
    
    for (class_name, data), color in zip(pr_data.items(), colors):
        ax.plot(data['recall'], data['precision'], color=color, lw=2,
                label=f'{class_name} (AP = {data["average_precision"]:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curves — {model_name}')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)


def plot_segmentation_grid(images, gradcams, pseudo_masks, pred_masks, overlays,
                           save_path, num_samples=4):
    """
    Create a visualization grid showing the full pipeline:
    Original → Grad-CAM → Pseudo Mask → Predicted Mask → Overlay
    
    Args:
        images: List of original images (H, W, 3) float [0, 1]
        gradcams: List of Grad-CAM heatmaps (H, W) float [0, 1]
        pseudo_masks: List of pseudo masks (H, W) binary
        pred_masks: List of predicted segmentation masks (H, W) float [0, 1]
        overlays: List of overlay images (H, W, 3) float [0, 1]
        save_path: path to save figure
        num_samples: number of sample rows
    """
    n = min(num_samples, len(images))
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    
    if n == 1:
        axes = axes[np.newaxis, :]
    
    col_titles = ['Original Image', 'Grad-CAM', 'Pseudo Mask (GT)', 'Predicted Mask', 'Overlay']
    
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight='bold')
    
    for i in range(n):
        axes[i, 0].imshow(images[i])
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gradcams[i], cmap='jet')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pseudo_masks[i], cmap='gray')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(pred_masks[i], cmap='gray')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(overlays[i])
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_model_comparison_table(comparison_data: List[Dict], save_path: str):
    """
    Generate a model comparison table as a figure.
    
    Args:
        comparison_data: List of dicts, each with keys:
            'model', 'accuracy', 'precision', 'recall', 'f1',
            'params_M', 'training_time_min'
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 3 + len(comparison_data) * 0.6))
    ax.axis('off')
    
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Params (M)', 'Time (min)']
    
    table_data = []
    for d in comparison_data:
        table_data.append([
            d.get('model', ''),
            f"{d.get('accuracy', 0):.4f}",
            f"{d.get('precision', 0):.4f}",
            f"{d.get('recall', 0):.4f}",
            f"{d.get('f1', 0):.4f}",
            f"{d.get('params_M', 0):.2f}",
            f"{d.get('training_time_min', 0):.1f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#D9E2F3' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(columns)):
            table[i, j].set_facecolor(color)
    
    # Highlight best F1 row
    if table_data:
        best_idx = max(range(len(table_data)), 
                       key=lambda i: float(table_data[i][4]))
        for j in range(len(columns)):
            table[best_idx + 1, j].set_facecolor('#C6EFCE')
    
    ax.set_title('Model Comparison Results', fontsize=14, fontweight='bold', pad=20)
    fig.savefig(save_path)
    plt.close(fig)


def plot_ablation_table(ablation_data: List[Dict], save_path: str):
    """
    Generate an ablation study table as a figure.
    
    Args:
        ablation_data: List of dicts, each with:
            'experiment', 'accuracy', 'f1', 'notes'
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 3 + len(ablation_data) * 0.6))
    ax.axis('off')
    
    columns = ['Experiment', 'Accuracy', 'F1-Score', 'Δ Accuracy', 'Notes']
    
    # Calculate deltas relative to baseline (first row)
    baseline_acc = ablation_data[0].get('accuracy', 0) if ablation_data else 0
    
    table_data = []
    for d in ablation_data:
        delta = d.get('accuracy', 0) - baseline_acc
        delta_str = f"{delta:+.4f}" if delta != 0 else "baseline"
        table_data.append([
            d.get('experiment', ''),
            f"{d.get('accuracy', 0):.4f}",
            f"{d.get('f1', 0):.4f}",
            delta_str,
            d.get('notes', '')
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    for j in range(len(columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(table_data) + 1):
        color = '#D9E2F3' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(columns)):
            table[i, j].set_facecolor(color)
    
    ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold', pad=20)
    fig.savefig(save_path)
    plt.close(fig)


def plot_severity_distribution(severities: List[float], class_names_per_sample: List[str],
                               save_path: str):
    """Plot severity distribution per disease class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    axes[0].hist(severities, bins=20, color='#4472C4', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Severity (%)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Disease Severity Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Per-class box plot
    unique_classes = sorted(set(class_names_per_sample))
    class_severities = [[s for s, c in zip(severities, class_names_per_sample) if c == cls] 
                         for cls in unique_classes]
    
    bp = axes[1].boxplot(class_severities, labels=unique_classes, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1].set_xlabel('Disease Class')
    axes[1].set_ylabel('Severity (%)')
    axes[1].set_title('Severity by Disease Class')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_multi_model_curves(all_histories: Dict[str, dict], save_dir: str):
    """
    Plot overlaid training curves for multiple models (for comparison).
    
    Args:
        all_histories: {model_name: history_dict}
        save_dir: directory to save
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
    # Validation accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, history), color in zip(all_histories.items(), colors):
        epochs = range(1, len(history['val_acc']) + 1)
        ax.plot(epochs, history['val_acc'], color=color, lw=2, 
                label=f'{name}', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Comparison Across Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    fig.savefig(save_dir / "multi_model_val_accuracy.png")
    plt.close(fig)
    
    # Validation loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, history), color in zip(all_histories.items(), colors):
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'], color=color, lw=2, 
                label=f'{name}', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison Across Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir / "multi_model_val_loss.png")
    plt.close(fig)
