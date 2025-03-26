import os
import random
import shutil
from pathlib import Path
# 필터링된 이미지를 가지고 rare disease에 대해 데이터 수 3100장으로 맞추기

# Define paths
base_dir = Path('/home/minelab/desktop/Jack/step_vet_train/datasets')
source_paths = {
    'blepharitis': base_dir / 'matching_filtered/blepharitis/filtered_images',
    'keratitis': base_dir / 'matching_filtered/keratitis/filtered_images'
}
origin_paths = {
    'blepharitis': base_dir / 'origin/blepharitis',
    'keratitis': base_dir / 'origin/keratitis'
}
dest_paths = {
    'blepharitis': base_dir / 'refined_matching/refined_dataset/blepharitis',
    'keratitis': base_dir / 'refined_matching/refined_dataset/keratitis'
}
num_images = {
    'blepharitis': 2150,
    'keratitis': 1908
}

def copy_random_images(source_dir, dest_dir, count):
    """Copy random images from source_dir to dest_dir"""
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all image files in the source directory (png, jpg, jpeg)
    image_extensions = ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG']
    all_images = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in [ext.lower() for ext in image_extensions]]
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

def copy_all_images(source_dir, dest_dir):
    """Copy all images from source_dir to dest_dir"""
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all image files in the source directory (png, jpg, jpeg)
    image_extensions = ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG']
    all_images = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in [ext.lower() for ext in image_extensions]]
    total_images = len(all_images)
    print(f"Found {total_images} images in {source_dir}")
    
    # Copy all images
    for img in all_images:
        src_path = os.path.join(source_dir, img)
        dst_path = os.path.join(dest_dir, img)
        shutil.copy2(src_path, dst_path)
    
    # Count files in destination directory after copy
    copied_count = len([f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))])
    print(f"Copied {len(all_images)} images from origin to {dest_dir}")
    print(f"Total files in {dest_dir}: {copied_count}")
    
    return copied_count

def main():
    results = {}
    
    # Process each category
    for category in ['blepharitis', 'keratitis']:
        print(f"\nProcessing {category}...")
        source_dir = source_paths[category]
        origin_dir = origin_paths[category]
        dest_dir = dest_paths[category]
        count = num_images[category]
        
        # First copy random filtered images
        filtered_count = copy_random_images(source_dir, dest_dir, count)
        
        # Then copy all images from origin directory
        print(f"\nCopying original {category} images...")
        origin_count = copy_all_images(origin_dir, dest_dir)
        
        # Store total count
        results[category] = len([f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))])
    
    # Print summary
    print("\n===== Summary =====")
    for category, count in results.items():
        print(f"{category}: {count} images")

if __name__ == "__main__":
    main() 