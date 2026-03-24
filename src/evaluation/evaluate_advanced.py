import torch
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model_factory import get_model
from src.data.data_loaders import get_advanced_dataloaders
from sklearn.preprocessing import label_binarize

def evaluate_advanced(model_name: str):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    BATCH_SIZE = config["batch_size"]
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    
    EXP_DIR = Path(config["experiments"][model_name])
    CHECKPOINT = EXP_DIR / "checkpoints" / "best_model.pth"
    
    if not CHECKPOINT.exists():
        print(f"No checkpoint found for {model_name} at {CHECKPOINT}")
        return

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only need test loader
    _, _, test_loader, details = get_advanced_dataloaders(BATCH_SIZE, IMG_SIZE, use_class_weights=False)
    classes = details["classes"]

    model = get_model(model_name, NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print(f"Evaluating {model_name} on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Classification Report (Precision, Recall, F1)
    report = classification_report(all_labels, all_preds, target_names=classes)
    print("\n--- Classification Report ---")
    print(report)
    with open(EXP_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(EXP_DIR / "confusion_matrix.png")
    plt.close()

    # ROC Curves and AUC
    y_bin = label_binarize(all_labels, classes=[i for i in range(NUM_CLASSES)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(EXP_DIR / "roc_curves.png")
    plt.close()
    
    print(f"Evaluation artifacts saved to {EXP_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet", help="Model name to evaluate")
    args = parser.parse_args()
    evaluate_advanced(args.model)
