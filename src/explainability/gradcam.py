import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import Tuple

def get_target_layer(model, model_name: str):
    """
    Returns the appropriate target layer for Grad-CAM depending on the architecture.
    """
    # Using simple heuristic, but this should be tailored if custom models are used.
    if "convnext" in model_name:
        return [model.stages[-1][-1]]
    elif "efficientnet" in model_name:
        if hasattr(model, 'conv_head'): # timm version
            return [model.conv_head]
        elif hasattr(model, 'features'): # torchvision version
            return [model.features[-1]]
        return [model.blocks[-1][-1]]
    elif "swin" in model_name:
        # For Swin Transformer, target the final norm layer or final block
        return [model.layers[-1].blocks[-1].norm2]
    elif "resnet" in model_name:
        if hasattr(model, 'model'): # ResNetFineTuned
            return [model.model.layer4[-1]]
        elif hasattr(model, 'layer4'): # torchvision resnet
            return [model.layer4[-1]]
    
    # Fallback to the last common module if possible
    raise ValueError(f"Grad-CAM target layer logic not defined for {model_name}")

def generate_gradcam_heatmap(model, model_name: str, input_tensor: torch.Tensor, original_img: np.ndarray, target_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap for a given input tensor and overlay it on the original image.

    Args:
        model: PyTorch model.
        model_name (str): Identifier for the model architecture to find target layer.
        input_tensor (torch.Tensor): Preprocessed input tensor (1, C, H, W).
        original_img (np.ndarray): Original image as a float32 numpy array in [0, 1] range (H, W, 3).
        target_class (int, optional): The class to explain. If None, explains highest scoring class.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The raw heatmap (H, W) and the overlaid image (H, W, 3) in [0, 1] float32 format.
    """
    target_layers = get_target_layer(model, model_name)
    
    # Construct the CAM object once, and then re-use it on many images
    # We use reshape_transform for Vision Transformers (Swin/ViT) if supported, but pytorch_grad_cam handles many automatically now
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=input_tensor.is_cuda)
    
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay on original image
    # Note: original_img must be float [0, 1]
    visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
    
    return grayscale_cam, visualization
