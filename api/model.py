import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import yaml
import os
import base64
import numpy as np

from src.models.model_factory import get_model
from src.explainability.gradcam import generate_gradcam_heatmap

CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    MODEL_NAME = config.get("deploy_model", "efficientnet")
    CHECKPOINT_PATH = config.get("checkpoints", {}).get(MODEL_NAME, "experiments/efficientnet/checkpoints/efficientnet_best.pth")
else:
    MODEL_NAME = "efficientnet"
    CHECKPOINT_PATH = "experiments/efficientnet/checkpoints/efficientnet_best.pth"

CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model() -> nn.Module:
    """
    Load the configured model and set it to evaluation mode.
    """
    model = get_model(MODEL_NAME, len(CLASS_NAMES), pretrained=False)
    
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using untrained model weights.")
        
    model.to(DEVICE)
    model.eval()
    return model

def preprocess(img_bytes: bytes) -> tuple:
    """
    Preprocess an image for the model and for Grad-CAM.
    """
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Original image array normalized to [0, 1] for Grad-CAM visualization
    img_resized = image.resize((256, 256))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor, img_arr

def predict(model: nn.Module, img_bytes: bytes) -> dict:
    """
    Predict class and probabilities from an image, and generate Grad-CAM heatmap.
    """
    x, img_arr = preprocess(img_bytes)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]

    # Generate Grad-CAM heatmap using the model with gradients temporarily enabled for the heatmap extraction
    gradcam_base64 = None
    try:
        _, vis = generate_gradcam_heatmap(model, MODEL_NAME, x, img_arr, target_class=pred_idx)
        # vis is RGB [0, 1] float32 image. Convert to uint8 BGR for encoding.
        vis_uint8 = (vis * 255).astype(np.uint8)
        # convert PIL image to base64
        import cv2
        vis_bgr = cv2.cvtColor(vis_uint8, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', vis_bgr)
        gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Grad-CAM error: {e}")

    return {
        "predicted_class": pred_class,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "gradcam_base64": gradcam_base64
    }