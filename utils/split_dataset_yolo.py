import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(
    source_dir,
    target_total=3100,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    """
    Split dataset into train, val, and test sets with specified ratios.
    Each class will have exactly target_total images.
    Test set will not include images starting with 'seed'.
    
    Args:
        source_dir (str): Source directory containing class folders
        target_total (int): Target total number of images per class
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create YOLO dataset structure
    base_dir = Path(source_dir).parent
    yolo_dir = base_dir / "yolo_dataset"
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    yolo_dir.mkdir(exist_ok=True)
    
    # Create train, val, test directories
    for split in ['train', 'val', 'test']:
        (yolo_dir / split).mkdir(exist_ok=True)
    
    # Calculate target counts for each split
    target_test = int(target_total * test_ratio)
    target_val = int(target_total * val_ratio)
    target_train = target_total - target_test - target_val
    
    # Process each class
    statistics = defaultdict(lambda: defaultdict(int))
    
    for class_dir in Path(source_dir).iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Create class directories
        for split in ['train', 'val', 'test']:
            (yolo_dir / split / class_name).mkdir(exist_ok=True)
        
        # Get all images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        seed_images = [img for img in images if img.name.startswith('seed')]
        non_seed_images = [img for img in images if not img.name.startswith('seed')]
        
        # Shuffle images
        random.shuffle(seed_images)
        random.shuffle(non_seed_images)
        
        # First allocate test images from non-seed images
        test_images = non_seed_images[:target_test]
        remaining_non_seed = non_seed_images[target_test:]
        
        # Calculate how many images we need for validation and training
        remaining_needed = target_total - target_test
        available_images = remaining_non_seed + seed_images
        
        if len(available_images) < remaining_needed:
            print(f"Warning: Not enough images for {class_name}. Need {remaining_needed} more but only have {len(available_images)}")
            # Use what we have
            val_count = int(len(available_images) * (val_ratio / (train_ratio + val_ratio)))
            train_count = len(available_images) - val_count
        else:
            # Use exactly what we need
            val_count = target_val
            train_count = target_train
        
        # Split remaining images between train and validation
        val_images = available_images[:val_count]
        train_images = available_images[val_count:val_count + train_count]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(img, yolo_dir / 'train' / class_name / img.name)
            statistics['train'][class_name] += 1
        
        for img in val_images:
            shutil.copy2(img, yolo_dir / 'val' / class_name / img.name)
            statistics['val'][class_name] += 1
        
        for img in test_images:
            shutil.copy2(img, yolo_dir / 'test' / class_name / img.name)
            statistics['test'][class_name] += 1
    
    # Print final statistics
    print("\n=== Dataset Statistics ===")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.capitalize()}:")
        total = 0
        for class_name in sorted(statistics[split].keys()):
            count = statistics[split][class_name]
            print(f"  - {class_name}: {count}")
            total += count
        print(f"  Total: {total}")

if __name__ == "__main__":
    source_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/refined_dataset"
    split_dataset(source_dir)
