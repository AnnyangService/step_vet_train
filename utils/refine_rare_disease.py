import os
import random
import shutil
from pathlib import Path
# 필터링된 이미지를 가지고 rare disease에 대해 데이터 수 3100장으로 맞추기

# Define paths
base_dir = Path('/home/minelab/desktop/Jack/step_vet_train/datasets')
source_paths = {
    'blepharitis': base_dir / 'filtered/blepharitis/filtered_images',
    'keratitis': base_dir / 'filtered/keratitis/filtered_images'
}
dest_paths = {
    'blepharitis': base_dir / 'dataset/blepharitis',
    'keratitis': base_dir / 'dataset/keratitis'
}
num_images = {
    'blepharitis': 2150,
    'keratitis': 1908
}

def copy_random_images(source_dir, dest_dir, count):
    """Copy random images from source_dir to dest_dir"""
    
    # Get all PNG images in the source directory
    all_images = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    total_images = len(all_images)
    print(f"Found {total_images} images in {source_dir}")
    
    # Select random images
    if count > total_images:
        print(f"Warning: Requested {count} images but only {total_images} are available. Using all images.")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, count)
    
    # Copy selected images
    for img in selected_images:
        src_path = os.path.join(source_dir, img)
        dst_path = os.path.join(dest_dir, img)
        shutil.copy2(src_path, dst_path)
    
    # Count files in destination directory
    copied_count = len([f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))])
    print(f"Copied {len(selected_images)} images to {dest_dir}")
    print(f"Total files in {dest_dir}: {copied_count}")
    
    return copied_count

def main():
    results = {}
    
    # Process each category
    for category in ['blepharitis', 'keratitis']:
        print(f"\nProcessing {category}...")
        source_dir = source_paths[category]
        dest_dir = dest_paths[category]
        count = num_images[category]
        
        results[category] = copy_random_images(source_dir, dest_dir, count)
    
    # Print summary
    print("\n===== Summary =====")
    for category, count in results.items():
        print(f"{category}: {count} images")

if __name__ == "__main__":
    main() 