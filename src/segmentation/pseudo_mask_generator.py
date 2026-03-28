import cv2
import numpy as np
import torch
from pathlib import Path

class PseudoMaskGenerator:
    """
    Generates binary segmentation masks from Grad-CAM heatmaps.
    This enables weakly supervised learning by using classification
    explanations as ground truth for a U-Net segmentation model.
    """
    def __init__(self, threshold: float = 0.5, kernel_size: int = 5):
        self.threshold = threshold
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def generate(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Convert a float32 heatmap [0, 1] into a binary mask.
        
        Args:
            heatmap (np.ndarray): Grad-CAM heatmap (H, W).
            
        Returns:
            np.ndarray: Binary mask (H, W) where 1 is diseased, 0 is background.
        """
        # 1. Thresholding
        binary_mask = (heatmap > self.threshold).astype(np.uint8)
        
        # 2. Morphological Operations to clean noise (Opening then Dilation)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel)
        binary_mask = cv2.dilate(binary_mask, self.kernel, iterations=1)
        
        return binary_mask

    def save_mask(self, mask: np.ndarray, save_path: str):
        """
        Saves the binary mask as an image (0-255 scaling).
        """
        cv2.imwrite(save_path, mask * 255)

def process_dataset_to_pseudo_masks(model, model_name, dataloader, device, output_dir, threshold=0.4):
    """
    Iterates through a classification dataloader, generates Grad-CAM heatmaps,
    converts them to pseudo-masks, and saves them for U-Net training.
    """
    from src.explainability.gradcam import generate_gradcam_heatmap
    
    output_path = Path(output_dir)
    img_dir = output_path / "images"
    mask_dir = output_path / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    generator = PseudoMaskGenerator(threshold=threshold)
    model.eval()
    
    print(f"Generating pseudo-masks using {model_name}...")
    
    count = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            
            for j in range(images.size(0)):
                img_tensor = images[j].unsqueeze(0)
                # Convert tensor to numpy for visualization overlay if needed, 
                # but for pseudo-mask we just need the raw heatmap.
                # Assuming image is normalized, we need it in [0, 1] for GradCAM overlay logic 
                # but generate_gradcam_heatmap handles the heatmap extraction.
                
                # We need a dummy original_img for the display logic in the existing function
                dummy_img = np.zeros((img_tensor.shape[2], img_tensor.shape[3], 3), dtype=np.float32)
                
                # Enable grads for heatmap generation even in eval mode
                with torch.set_grad_enabled(True):
                    heatmap, _ = generate_gradcam_heatmap(model, model_name, img_tensor, dummy_img, target_class=labels[j].item())
                
                # Generate binary mask
                pseudo_mask = generator.generate(heatmap)
                
                # Save
                filename = f"sample_{count}.png"
                # Save original image (denormalize if necessary, but here we save as is or re-read)
                # For simplicity, we assume the dataset images will be copied/aligned elsewhere 
                # or we save the tensor here.
                
                mask_save_path = str(mask_dir / filename)
                generator.save_mask(pseudo_mask, mask_save_path)
                
                count += 1
                
    print(f"Finished. Generated {count} pseudo-masks in {output_dir}")
