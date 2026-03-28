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

def swin_reshape_transform(tensor, height=7, width=7):
    """
    Reshape Swin Transformer output sequence (B, N, C) back to 2D spatial (B, C, H, W).
    Default for swin_tiny is 224/32 = 7x7 spatial grid at the end.
    """
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def generate_gradcam_heatmap(model, model_name: str, input_tensor: torch.Tensor, original_img: np.ndarray, target_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap for a given input tensor and overlay it on the original image.
    """
    target_layers = get_target_layer(model, model_name)
    
    # Swin/ViT specific transform
    reshape_transform = None
    if "swin" in model_name:
        reshape_transform = swin_reshape_transform
    
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay on original image
    visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
    
    return grayscale_cam, visualization
