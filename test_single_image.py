import argparse
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from src.models.model_factory import get_model
from api.model import preprocess, predict, CLASS_NAMES
import yaml

def test_image(image_path: str, model_name: str = None):
    """
    Test a single image using the specified model and display the Grad-CAM heatmap.
    """
    if not Path(image_path).exists():
        print(f"Error: Image '{image_path}' not found.")
        return

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if model_name is None:
        model_name = config.get("deploy_model", "efficientnet")

    checkpoint_path = config.get("checkpoints", {}).get(model_name)
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint '{checkpoint_path}' not found for model '{model_name}'. Using random weights.")

    print(f"Loading model '{model_name}'...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Processing image '{image_path}'...")
    # Read image bytes for the existing api prediction logic
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Get predictions
    result = predict(model, img_bytes)
    
    predicted_class = result["predicted_class"]
    probabilities = result["probabilities"]
    gradcam_b64 = result["gradcam_base64"]

    print(f"\n--- Prediction Results ---")
    print(f"Predicted Class: {predicted_class}")
    print("Probabilities:")
    for cls_name, prob in probabilities.items():
        print(f"  {cls_name}: {prob:.4f}")

    if gradcam_b64:
        import base64
        import numpy as np
        
        # Decode Grad-CAM
        img_data = base64.b64decode(gradcam_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        heatmap = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Original Image
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(original)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(heatmap)
        ax[1].set_title(f"Grad-CAM (Predicted: {predicted_class})")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("\nGrad-CAM generation failed or is unsupported for this model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a single image for Sugarcane Disease Detection.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--model", type=str, default=None, help="Model architecture (e.g., efficientnet, convnext, swin)")
    args = parser.parse_args()

    test_image(args.image_path, args.model)
