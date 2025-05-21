import os
import shutil
import random
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def get_random_files(directory, num_files):
    """Get random files from a directory"""
    files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
    if len(files) < num_files:
        raise ValueError(f"Not enough files in {directory}. Required: {num_files}, Available: {len(files)}")
    return random.sample(files, num_files)

def copy_files(source_files, source_dir, target_dir, class_name):
    """Copy files to target directory with class name"""
    for file in source_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(target_dir, class_name, file)
        shutil.copy2(src, dst)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Base directories
    base_dir = Path("/home/minelab/desktop/Jack/step_vet_train/datasets")
    origin_dir = base_dir / "origin"
    filtered_dir = base_dir / "matching_filtered/blepharitis/filtered_images"
    
    # Create output directories
    output_base = base_dir / "step2"
    splits = ['train', 'val', 'test']
    classes = ['other', 'corneal', 'blepharitis']
    
    for split in splits:
        for class_name in classes:
            create_directory(output_base / split / class_name)
    
    # Sample files for 'other' class (700 from each source)
    other_sources = ['keratitis', '결막염', '각막궤양']
    other_files = []
    other_source_map = {}  # Map to keep track of which file came from which source
    
    for source in other_sources:
        source_dir = origin_dir / source
        files = get_random_files(source_dir, 700)
        other_files.extend(files)
        for file in files:
            other_source_map[file] = source
    
    # Sample files for 'corneal' class
    corneal_dir = origin_dir / '각막부골편'
    corneal_files = get_random_files(corneal_dir, 2100)
    
    # Get all blepharitis files
    blepharitis_dir = origin_dir / 'blepharitis'
    blepharitis_files = [f for f in os.listdir(blepharitis_dir) if f.endswith('.jpg')]
    
    # Get additional blepharitis files from filtered images
    filtered_files = [f for f in os.listdir(filtered_dir) if f.endswith('.png')]
    additional_blepharitis = get_random_files(filtered_dir, 1150)
    blepharitis_files.extend(additional_blepharitis)
    
    # Split data into train/val/test
    # First, separate blepharitis seed files
    blepharitis_seed_files = [f for f in blepharitis_files if f.startswith('seed')]
    blepharitis_non_seed_files = [f for f in blepharitis_files if not f.startswith('seed')]
    
    # Split other and corneal classes
    random.shuffle(other_files)
    random.shuffle(corneal_files)
    
    # Split blepharitis files (ensuring no seed files in val/test)
    random.shuffle(blepharitis_non_seed_files)
    random.shuffle(blepharitis_seed_files)
    
    # Calculate split sizes
    val_size = test_size = 150
    train_size_other = len(other_files) - val_size - test_size
    train_size_corneal = len(corneal_files) - val_size - test_size
    train_size_blepharitis = len(blepharitis_files) - val_size - test_size
    
    # Split the data
    splits = {
        'train': {
            'other': other_files[:train_size_other],
            'corneal': corneal_files[:train_size_corneal],
            'blepharitis': blepharitis_seed_files + blepharitis_non_seed_files[:train_size_blepharitis - len(blepharitis_seed_files)]
        },
        'val': {
            'other': other_files[train_size_other:train_size_other + val_size],
            'corneal': corneal_files[train_size_corneal:train_size_corneal + val_size],
            'blepharitis': blepharitis_non_seed_files[train_size_blepharitis - len(blepharitis_seed_files):train_size_blepharitis - len(blepharitis_seed_files) + val_size]
        },
        'test': {
            'other': other_files[train_size_other + val_size:],
            'corneal': corneal_files[train_size_corneal + val_size:],
            'blepharitis': blepharitis_non_seed_files[train_size_blepharitis - len(blepharitis_seed_files) + val_size:train_size_blepharitis - len(blepharitis_seed_files) + val_size + test_size]
        }
    }
    
    # Copy files to their respective directories
    for split in splits:
        for class_name, files in splits[split].items():
            if class_name == 'blepharitis':
                # Handle both origin and filtered images
                for file in files:
                    if file.startswith('seed'):
                        src = os.path.join(filtered_dir, file)
                    else:
                        src = os.path.join(blepharitis_dir, file)
                    dst = os.path.join(output_base, split, class_name, file)
                    shutil.copy2(src, dst)
            elif class_name == 'other':
                # Handle other class files from different sources
                for file in files:
                    source = other_source_map[file]
                    src = os.path.join(origin_dir, source, file)
                    dst = os.path.join(output_base, split, class_name, file)
                    shutil.copy2(src, dst)
            else:  # corneal class
                source_dir = origin_dir / '각막부골편'
                copy_files(files, source_dir, output_base / split, class_name)

if __name__ == "__main__":
    main()
