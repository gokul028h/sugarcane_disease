"""
Unified metrics computation for classification and segmentation tasks.
Supports per-class metrics, IoU, Dice coefficient, and aggregated statistics.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(y_true, y_pred, y_probs, class_names):
    """
    Compute comprehensive classification metrics.
    
    Returns:
        dict with accuracy, per-class precision/recall/f1, confusion matrix,
        ROC data, PR data, and aggregated metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    # Weighted and macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Binarize for ROC/PR curves
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    y_probs = np.array(y_probs)
    
    # ROC curves
    roc_data = {}
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[class_names[i]] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc)
        }
    
    # PR curves
    pr_data = {}
    for i in range(len(class_names)):
        prec_curve, rec_curve, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        pr_data[class_names[i]] = {
            "precision": prec_curve.tolist(),
            "recall": rec_curve.tolist(),
            "average_precision": float(ap)
        }
    
    # Text report
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    return {
        "accuracy": float(accuracy),
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            } for i in range(len(class_names))
        },
        "macro_avg": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1_score": float(f1_macro)
        },
        "weighted_avg": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1_score": float(f1_weighted)
        },
        "confusion_matrix": cm.tolist(),
        "roc_data": roc_data,
        "pr_data": pr_data,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict
    }


def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    
    Args:
        pred_mask: Predicted mask (float, 0-1 range or binary)
        gt_mask: Ground truth mask (float, 0-1 range or binary)
        threshold: Binarization threshold
    
    Returns:
        float: IoU score
    """
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > threshold).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    if union == 0:
        return 1.0  # Both empty
    
    return float(intersection / union)


def compute_dice(pred_mask, gt_mask, threshold=0.5, smooth=1e-6):
    """
    Compute Dice coefficient for binary segmentation.
    
    Args:
        pred_mask: Predicted mask (float, 0-1 range or binary)
        gt_mask: Ground truth mask (float, 0-1 range or binary)
        threshold: Binarization threshold
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        float: Dice coefficient
    """
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > threshold).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    
    dice = (2.0 * intersection + smooth) / (np.sum(pred_binary) + np.sum(gt_binary) + smooth)
    return float(dice)


def compute_segmentation_metrics(pred_masks, gt_masks, threshold=0.5):
    """
    Compute batch segmentation metrics (IoU and Dice).
    
    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        threshold: Binarization threshold
    
    Returns:
        dict with mean IoU, mean Dice, and per-sample values
    """
    ious = []
    dices = []
    
    for pred, gt in zip(pred_masks, gt_masks):
        ious.append(compute_iou(pred, gt, threshold))
        dices.append(compute_dice(pred, gt, threshold))
    
    return {
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
        "per_sample_iou": ious,
        "per_sample_dice": dices
    }
