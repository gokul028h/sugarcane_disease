from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple
import yaml
import os

CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

BATCH_SIZE = config.get("batch_size", 32)
IMG_SIZE = config.get("input_size", 256)

DATA_PATHS = {
    "train": "data/train",
    "val": "data/val",
    "test": "data/test"
}

def get_dataloaders(batch_size: int = BATCH_SIZE, img_size: int = IMG_SIZE
                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns PyTorch DataLoaders for training, validation, and testing datasets using standard transforms.
    Maintained for backwards compatibility.
    """

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=DATA_PATHS["train"], transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=DATA_PATHS["val"], transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=DATA_PATHS["test"], transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_advanced_dataloaders(batch_size: int = BATCH_SIZE, img_size: int = IMG_SIZE, use_class_weights: bool = True
                   ) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Returns data loaders with RandAugment, AutoAugment, and WeightedRandomSampler.
    Also returns a dictionary of dataset details like class weights.
    """
    import torch
    from torchvision.transforms import v2

    train_transforms = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=DATA_PATHS["train"], transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=DATA_PATHS["val"], transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=DATA_PATHS["test"], transform=val_test_transforms)

    sampler = None
    class_weights_tensor = None

    if use_class_weights:
        import numpy as np
        target_list = train_dataset.targets
        class_counts = np.bincount(target_list)
        class_weights = 1. / class_counts
        weights = class_weights[target_list]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=True
        )
        
        # calculate weights for CrossEntropyLoss
        class_weights_tensor = torch.FloatTensor(len(train_dataset.samples) / (len(class_counts) * class_counts))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    details = {
        "class_weights": class_weights_tensor,
        "classes": train_dataset.classes,
        "num_classes": len(train_dataset.classes)
    }

    return train_loader, val_loader, test_loader, details

def get_mixup_cutmix(num_classes: int = 6, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0):
    """
    Returns a v2 Transform that applies CutMix or MixUp randomly.
    Should be applied to the batch (images, targets) inside the training loop.
    """
    from torchvision.transforms import v2
    cutmix = v2.CutMix(num_labels=num_classes, alpha=cutmix_alpha)
    mixup = v2.MixUp(num_labels=num_classes, alpha=mixup_alpha)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    return cutmix_or_mixup