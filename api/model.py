import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import yaml
import os
import base64
import numpy as np
import cv2

from src.models.model_factory import get_model
from src.models.segmentation_models import UNet
from src.explainability.gradcam import generate_gradcam_heatmap
from src.evaluation.severity import calculate_severity

CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    MODEL_NAME = config.get("deploy_model", "efficientnet")
    CLS_CHECKPOINT = config.get("checkpoints", {}).get(MODEL_NAME, f"experiments/{MODEL_NAME}/checkpoints/{MODEL_NAME}_best.pth")
    SEG_CHECKPOINT = config.get("checkpoints", {}).get("unet_segmentation", "experiments/unet_segmentation/checkpoints/unet_best.pth")
else:
    MODEL_NAME = "efficientnet"
    CLS_CHECKPOINT = "experiments/efficientnet/checkpoints/efficientnet_best.pth"
    SEG_CHECKPOINT = "experiments/unet_segmentation/checkpoints/unet_best.pth"

CLASS_NAMES = ["BacterialBlights", "Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiStagePipeline:
    def __init__(self):
        # 1. Load Classifier
        self.cls_model = get_model(MODEL_NAME, len(CLASS_NAMES), pretrained=False)
        if os.path.exists(CLS_CHECKPOINT):
            self.cls_model.load_state_dict(torch.load(CLS_CHECKPOINT, map_location=DEVICE))
        self.cls_model.to(DEVICE)
        self.cls_model.eval()

        # 2. Load Segmenter (U-Net)
        self.seg_model = UNet(n_channels=3, n_classes=1).to(DEVICE)
        if os.path.exists(SEG_CHECKPOINT):
            self.seg_model.load_state_dict(torch.load(SEG_CHECKPOINT, map_location=DEVICE))
        self.seg_model.to(DEVICE)
        self.seg_model.eval()

def load_model():
    """
    Load the MultiStagePipeline into app state.
    """
    return MultiStagePipeline()

def preprocess(img_bytes: bytes) -> tuple:
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    img_resized = image.resize((256, 256))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor, img_arr

def encode_base64_img(img_array: np.ndarray) -> str:
    """Convert numpy HxWxC (RGB) array to base64 jpg string."""
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        # Grayscale mask
        vis_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        # RGB
        vis_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
    _, buffer = cv2.imencode('.jpg', vis_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def predict(pipeline: MultiStagePipeline, img_bytes: bytes) -> dict:
    x, img_arr = preprocess(img_bytes)

    # --- Classification Stage ---
    with torch.no_grad():
        outputs = pipeline.cls_model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]

    # --- Grad-CAM Explainability ---
    gradcam_base64 = None
    try:
        _, vis = generate_gradcam_heatmap(pipeline.cls_model, MODEL_NAME, x, img_arr, target_class=pred_idx)
        vis_uint8 = (vis * 255).astype(np.uint8)
        gradcam_base64 = encode_base64_img(vis_uint8)
    except Exception:
        pass

    # --- Segmentation Stage ---
    segmentation_base64 = None
    severity = 0.0
    if pred_class != "Healthy":
        with torch.no_grad():
            seg_out = pipeline.seg_model(x)
            seg_probs = torch.sigmoid(seg_out).cpu().numpy()[0, 0]
            
            # Binary mask visualization (yellow/red overlay mapped on black for simplicity)
            binary_mask = (seg_probs > 0.5).astype(np.uint8)
            
            # Encode visual mask
            overlay_mask = (binary_mask * 255)
            segmentation_base64 = encode_base64_img(overlay_mask)
            
            # Severity
            severity = calculate_severity(binary_mask)
    else:
        # If healthy, no disease to segment/quantify
        severity = 0.0

    return {
        "predicted_class": pred_class,
        "severity_percentage": severity,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "gradcam_base64": gradcam_base64,
        "segmentation_base64": segmentation_base64
    }