import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset_cv(
    source_dir,
    target_total=3100,
    class_targets=None,
    seed=42
):
    """
    Split dataset into train and test sets for cross validation.
    Each class will be split according to 9:1 ratio based on class_targets.
    Seed images will only be used in training set.
    
    Args:
        source_dir (str): Source directory containing class folders
        target_total (int): Default target number of images per class (only used if class_targets is None)
        class_targets (dict, optional): Dictionary mapping class names to target counts
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create dataset structure
    dataset_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset_cv"
    dataset_dir = Path(dataset_dir)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(exist_ok=True)
    
    # Initialize class_targets if not provided
    if class_targets is None:
        class_targets = {}
    
    # Create train directory (for cross validation)
    (dataset_dir / 'train').mkdir(exist_ok=True)
    
    # Process each class
    statistics = defaultdict(lambda: defaultdict(int))
    
    for class_dir in Path(source_dir).iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get target count for this class
        target_count = class_targets.get(class_name, target_total)
        
        # Calculate split sizes (9:1 ratio)
        target_test = target_count // 10
        target_train = target_count - target_test
        
        print(f"Target split for {class_name}:")
        print(f"  Total: {target_count}")
        print(f"  Train: {target_train}, Test: {target_test}")
        
        # Get all images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        seed_images = [img for img in images if img.name.startswith('seed')]
        non_seed_images = [img for img in images if not img.name.startswith('seed')]
        
        # Create class directory
        (dataset_dir / 'train' / class_name).mkdir(exist_ok=True)
        
        # Shuffle images
        random.shuffle(non_seed_images)
        random.shuffle(seed_images)
        
        # Split non-seed images according to target sizes
        test_images = non_seed_images[:target_test]
        train_images = non_seed_images[target_test:target_test + target_train]
        
        # If we need more images for train, use seed images
        if len(train_images) < target_train and seed_images:
            needed_train = target_train - len(train_images)
            train_images.extend(seed_images[:needed_train])
        
        # Check if we still don't have enough images
        if len(train_images) < target_train:
            print(f"Warning: Not enough images for {class_name}.")
            print(f"  - Train: {len(train_images)}/{target_train}")
            print(f"  - Test: {len(test_images)}/{target_test}")
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(img, dataset_dir / 'train' / class_name / img.name)
            statistics['train'][class_name] += 1
        
        for img in test_images:
            shutil.copy2(img, dataset_dir / 'train' / class_name / img.name)
            statistics['test'][class_name] += 1
    
    # Print final statistics
    print("\n=== Dataset Statistics ===")
    for split in ['train', 'test']:
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
        "blepharitis": 1400,
        "keratitis": 1400,
        "normal": 1400,
        "각막궤양": 1400,
        "각막부골편": 1400,
        "결막염": 1400
    }
    
    split_dataset_cv(source_dir, class_targets=class_targets) 