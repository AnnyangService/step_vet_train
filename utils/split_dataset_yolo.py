import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(
    source_dir,
    target_total=3100,
    class_targets=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    fixed_test_count=None,
    seed=42
):
    """
    Split dataset into train, val, and test sets with specified ratios.
    Each class will have either the class-specific target count or the default target_total images.
    All classes will have the same number of test samples if fixed_test_count is provided.
    Test set will not include images starting with 'seed'.
    
    Args:
        source_dir (str): Source directory containing class folders
        target_total (int): Default target number of images per class
        class_targets (dict, optional): Dictionary mapping class names to target counts
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set (ignored if fixed_test_count is provided)
        fixed_test_count (int, optional): Fixed number of test samples for each class
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create YOLO dataset structure
    yolo_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/yolo_dataset"
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    yolo_dir.mkdir(exist_ok=True)
    
    # Initialize class_targets if not provided
    if class_targets is None:
        class_targets = {}
    
    # Create train, val, test directories
    for split in ['train', 'val', 'test']:
        (yolo_dir / split).mkdir(exist_ok=True)
    
    # Process each class
    statistics = defaultdict(lambda: defaultdict(int))
    
    # If fixed_test_count is not provided, calculate it from the smallest class target
    if fixed_test_count is None:
        # Get all class names first
        class_names = [d.name for d in Path(source_dir).iterdir() if d.is_dir()]
        
        # Get the smallest target count
        smallest_target = float('inf')
        for class_name in class_names:
            class_target = class_targets.get(class_name, target_total)
            smallest_target = min(smallest_target, class_target)
        
        # Calculate test count based on the smallest class
        fixed_test_count = int(smallest_target * test_ratio)
        print(f"Using fixed test count: {fixed_test_count} for all classes")
    
    for class_dir in Path(source_dir).iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get class-specific target or use default
        class_target = class_targets.get(class_name, target_total)
        
        # Use fixed test count for all classes
        target_test = fixed_test_count
        
        # Calculate remaining for train and val
        remaining_for_train_val = class_target - target_test
        target_val = int(remaining_for_train_val * (val_ratio / (train_ratio + val_ratio)))
        target_train = remaining_for_train_val - target_val
        
        print(f"Target for {class_name}: {class_target} (train: {target_train}, val: {target_val}, test: {target_test})")
        
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
        remaining_needed = class_target - target_test
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
    source_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/refined_matching/refined_dataset"
    
    class_targets = {
        "blepharitis": 3100,
        "keratitis": 3100,
        "normal": 3100,
        "각막궤양": 3100,
        "각막부골편": 3100,
        "결막염": 3100
    }
    
    fixed_test_count = 310  # For example, 310 test images per class
    
    split_dataset(source_dir, class_targets=class_targets, fixed_test_count=fixed_test_count)
    
