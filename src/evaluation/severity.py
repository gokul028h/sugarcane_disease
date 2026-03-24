import numpy as np

def calculate_severity(binary_mask: np.ndarray) -> float:
    """
    Calculates the severity of the leaf disease based on a binary mask.
    
    Args:
        binary_mask (np.ndarray): Binary segmentation mask (1 for disease, 0 for background).
        
    Returns:
        float: Percentage of the leaf affected by the disease. 
               (diseased_pixels / total_pixels) * 100
    """
    # Assumes the mask is 2D (H, W).
    diseased_pixels = np.sum(binary_mask > 0)
    total_pixels = binary_mask.size
    
    if total_pixels == 0:
        return 0.0
        
    severity_percentage = (diseased_pixels / total_pixels) * 100.0
    return round(severity_percentage, 2)
