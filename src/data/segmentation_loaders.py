import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob

class SegmentationDataset(Dataset):
    """
    Dataset loader for Image + Binary Mask pairs.
    """
    def __init__(self, images_dir: str, masks_dir: str, img_size: int = 256, is_train: bool = True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.is_train = is_train

        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")))
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        
        # Mask loader - nearest exact pixels
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Masks always saved as .png in generate_pseudo_masks.py
        mask_path = os.path.join(self.masks_dir, f"{base_name}.png")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Grayscale
        
        # Augmentations for training
        if self.is_train:
            import random
            import torchvision.transforms.functional as TF
            
            # Since masks and images MUST map to same coordinates, apply identical flips manually.
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # ToTensor converts image to [0, 1] float.
        image_tensor = self.transform(image)
        
        # Mask tensor will be 0.0 or 1.0 roughly
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor

def get_segmentation_dataloaders(images_dir: str, masks_dir: str, batch_size: int = 16, img_size: int = 256, val_split: float = 0.2):
    """
    Returns (train_loader, val_loader) by splitting the dataset.
    """
    dataset = SegmentationDataset(images_dir, masks_dir, img_size=img_size, is_train=True)
    
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # We must use a deterministic random split for consistency
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    # Disable manual augmentation for validation explicitly by overriding the wrap attribute
    val_dataset.dataset.is_train = False 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
