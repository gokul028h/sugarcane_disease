import os
import random
import shutil
from tqdm import tqdm

# Source path
source_dir = r"E:\Downloads\dataset_sugarcane\Sugarcane_leafs"

# Destination path
dest_dir = r"C:\Users\HP\Desktop\desktop\sugarcane_disease\data"

# Classes
classes = ["BacterialBlights", "Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Image extensions
extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Fix randomness (optional)
random.seed(42)

print("\n========== DATASET SUMMARY ==========\n")

dataset_summary = {}

# Step 1: Count + Split Info
for cls in classes:
    class_path = os.path.join(source_dir, cls)
    
    images = [img for img in os.listdir(class_path) if img.lower().endswith(extensions)]
    total = len(images)
    
    train_count = int(train_ratio * total)
    val_count = int(val_ratio * total)
    test_count = total - train_count - val_count
    
    dataset_summary[cls] = {
        "total": total,
        "train": train_count,
        "val": val_count,
        "test": test_count,
        "images": images
    }
    
    print(f"{cls}: Total={total} | Train={train_count} | Val={val_count} | Test={test_count}")

print("\n====================================\n")

# Step 2: Copy with Progress Bar
for cls in classes:
    class_path = os.path.join(source_dir, cls)
    images = dataset_summary[cls]["images"]
    
    random.shuffle(images)
    
    train_end = dataset_summary[cls]["train"]
    val_end = train_end + dataset_summary[cls]["val"]
    
    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }
    
    for split_type, img_list in splits.items():
        dest_class_path = os.path.join(dest_dir, split_type, cls)
        os.makedirs(dest_class_path, exist_ok=True)
        
        print(f"\n📁 Copying {cls} → {split_type} ({len(img_list)} images)")
        
        for img in tqdm(img_list, desc=f"{cls}-{split_type}", unit="img"):
            src = os.path.join(class_path, img)
            dst = os.path.join(dest_class_path, img)
            shutil.copy2(src, dst)

print("\n✅ Dataset split completed successfully with progress tracking!")