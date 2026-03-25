import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import cv2
import torch
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from src.models.model_factory import get_model
from src.explainability.gradcam import get_target_layer

def generate_masks():
    """
    Generate binary pseudo-masks for the entire training dataset using Grad-CAM heatmaps
    from the pre-trained classification model.
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    IMAGE_DIR = config["data_paths"]["train"]
    OUTPUT_MASK_DIR = config["data_paths"]["seg_masks"]
    OUTPUT_IMG_DIR = config["data_paths"]["seg_images"]
    
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = config.get("deploy_model", "efficientnet")
    CHECKPOINT_PATH = config.get("checkpoints", {}).get(MODEL_NAME)
    NUM_CLASSES = config["num_classes"]

    # Load Model
    print(f"Loading {MODEL_NAME} for pseudo-mask generation...")
    model = get_model(MODEL_NAME, num_classes=NUM_CLASSES)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Warning: No trained checkpoint found. Using untrained model for mask generation! Checkpoints path:", CHECKPOINT_PATH)
    
    model.to(DEVICE)
    model.eval()

    target_layers = get_target_layer(model, MODEL_NAME)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Re-use GradCAM 
    cam = GradCAM(model=model, target_layers=target_layers)

    def process_image(img_path):
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil) / 255.0

        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # We don't specify target class, it uses highest scoring output implicitly
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        # Resize to original
        cam_resized = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))

        # Min-Max Normalize
        if cam_resized.max() > cam_resized.min():
            cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())

        # Thresholding (You can tune this, 0.5 is standard)
        threshold = 0.5
        mask = (cam_resized > threshold).astype(np.uint8) * 255

        # Improve mask quality via morphological operations to remove noise holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    print("Beginning dataset parsing...")
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                
                try:
                    mask, img_bgr = process_image(img_path)
                    
                    # Ensure matching filenames between images and masks
                    base_name = file.rsplit('.', 1)[0]
                    mask_save_path = os.path.join(OUTPUT_MASK_DIR, f"{base_name}.png")
                    img_save_path = os.path.join(OUTPUT_IMG_DIR, f"{base_name}.jpg")
                    
                    cv2.imwrite(mask_save_path, mask)
                    cv2.imwrite(img_save_path, img_bgr)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print(f"Pseudo masks successfully generated in {OUTPUT_MASK_DIR}")
    print(f"Source images copied to {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    generate_masks()
