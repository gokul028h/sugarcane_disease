import timm
import torch.nn as nn

def create_advanced_model(model_name: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Creates an advanced benchmark model using the `timm` library.
    
    Supported model_name options:
    - convnext: 'convnext_tiny'
    - efficientnet: 'efficientnet_b4'
    - swin: 'swin_tiny_patch4_window7_224'
    """
    
    valid_models = {
        'convnext': 'convnext_tiny',
        'efficientnet': 'efficientnet_b4',
        'swin': 'swin_tiny_patch4_window7_224'
    }
    
    if model_name not in valid_models:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(valid_models.keys())}")
        
    timm_model_name = valid_models[model_name]
    
    model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def load_legacy_model(model_name: str, num_classes: int = 5):
    """
    Loads legacy models (baseline_cnn, resnet_transfer, resnet50_finetuned)
    for backward compatibility and comparison.
    """
    if model_name == 'baseline_cnn':
        from .baseline_cnn import BaselineCNN
        return BaselineCNN(num_classes=num_classes)
    elif model_name == 'resnet_frozen':
        from .resnet_transfer import ResNetTransfer
        return ResNetTransfer(num_classes=num_classes)
    elif model_name == 'resnet_finetuned':
        from .resnet50_finetuned import ResNet50FineTuned
        return ResNet50FineTuned(num_classes=num_classes)
    else:
        raise ValueError(f"Legacy model {model_name} not found.")

def get_model(model_name: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Unified model factory to load both advanced and legacy models.
    """
    legacy_models = ['baseline_cnn', 'resnet_frozen', 'resnet_finetuned']
    if model_name in legacy_models:
        return load_legacy_model(model_name, num_classes)
    else:
        return create_advanced_model(model_name, num_classes, pretrained)
