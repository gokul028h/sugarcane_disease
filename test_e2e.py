import argparse
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import base64
import numpy as np

# Assuming models are initialized properly via load_model (or fake weights if untrained)
from api.model import load_model, predict

def run_pipeline(image_path: str):
    """
    Execute the entire classification -> segmentation -> severity pipeline on a single image.
    """
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found.")
        return

    print("Initializing Multi-Stage Pipeline...")
    pipeline = load_model()

    print(f"Processing image: {image_path}")
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # E2E Inference
    res = predict(pipeline, img_bytes)

    # Console Output
    print("\n" + "="*40)
    print("📝 IEEE MULTI-STAGE INFERENCE RESULT")
    print("="*40)
    print(f"Disease Classification : {res['predicted_class']}")
    print(f"Severity Estimation    : {res['severity_percentage']}%")
    print("-" * 40)
    print("Class Probabilities:")
    for cls, prob in res['probabilities'].items():
        print(f"  {cls}: {prob:.4f}")
    
    # Visualization
    num_plots = 1
    if res['gradcam_base64']: num_plots += 1
    if res['segmentation_base64']: num_plots += 1
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    
    # 1. Original
    orig_bgr = cv2.imread(image_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    idx = 1
    # 2. GradCAM
    if res['gradcam_base64']:
        arr = np.frombuffer(base64.b64decode(res['gradcam_base64']), np.uint8)
        cam = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(cam)
        axes[idx].set_title("Grad-CAM Focus")
        axes[idx].axis('off')
        idx += 1

    # 3. Segmentation Mask
    if res['segmentation_base64']:
        arr = np.frombuffer(base64.b64decode(res['segmentation_base64']), np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        
        # Overlay mask on image for presentation
        overlay = orig_rgb.copy()
        mask_resized = cv2.resize(mask, (orig_rgb.shape[1], orig_rgb.shape[0]))
        overlay[mask_resized > 128] = [255, 0, 0] # Highlight disease in red
        
        # Blend
        blended = cv2.addWeighted(orig_rgb, 0.5, overlay, 0.5, 0)
        
        axes[idx].imshow(blended)
        axes[idx].set_title(f"Segmentation Map ({res['severity_percentage']}%)")
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Multistage Pipeline CLI")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    run_pipeline(args.image_path)
