from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import os
import json
import shutil
from scipy.spatial.distance import cosine
import time
from tqdm import tqdm

def filter_by_class_dissimilarity(
    filtered_vector_path: str,
    other_class_vector_paths: Dict[str, str],
    filtered_images_dir: str,
    output_dir: str,
    keep_ratio: float = 0.5
):
    """
    Filter images by keeping only those with low similarity to other classes
    
    Args:
        filtered_vector_path: Path to filtered image vectors (.npy)
        other_class_vector_paths: Dictionary of other class vector paths {class_name: vector_path}
        filtered_images_dir: Directory containing filtered images
        output_dir: Directory to save results
        keep_ratio: Ratio of images to keep (0-1)
    """
    print(f"\n=== Filtering Images with Low Similarity to Other Classes (Keeping {keep_ratio*100:.0f}%) ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check vector files exist
    if not os.path.exists(filtered_vector_path):
        raise FileNotFoundError(f"Filtered image vector file not found: {filtered_vector_path}")
    
    # Check other class vector files exist
    for class_name, vector_path in other_class_vector_paths.items():
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Vector file for class {class_name} not found: {vector_path}")
    
    # Load vectors
    print(f"Loading vector files...")
    filtered_vectors = np.load(filtered_vector_path, allow_pickle=True).item()
    
    # Load other class vectors
    other_class_vectors = {}
    for class_name, vector_path in other_class_vector_paths.items():
        other_class_vectors[class_name] = np.load(vector_path, allow_pickle=True).item()
        print(f"Loaded: {len(other_class_vectors[class_name])} vectors (class {class_name})")
    
    print(f"Loaded: {len(filtered_vectors)} filtered image vectors")
    
    # Calculate maximum similarity to other classes for each filtered image
    print("Calculating similarity to other classes...")
    image_similarities = []
    
    for img_name, img_vector in tqdm(filtered_vectors.items(), desc="Calculating similarity"):
        # Maximum similarity to each class
        max_similarities = {}
        for class_name, class_vectors in other_class_vectors.items():
            max_sim = calculate_max_similarity(img_vector, class_vectors)
            max_similarities[class_name] = max_sim
        
        # Maximum similarity to any other class
        max_other_sim = max(max_similarities.values()) if max_similarities else 0
        
        # Extract image filename (remove path if present)
        img_filename = os.path.basename(img_name)
        
        # Store result
        image_similarities.append((img_name, img_filename, max_other_sim))
    
    # Sort by similarity to other classes (ascending - lower similarity is better)
    image_similarities.sort(key=lambda x: x[2])
    
    # Keep only top N% images
    keep_count = max(1, int(len(image_similarities) * keep_ratio))
    selected_images = image_similarities[:keep_count]
    
    print(f"Selected: {len(selected_images)} images (lowest similarity to other classes)")
    
    # Copy selected images
    output_images_dir = os.path.join(output_dir, "low_sim_images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Save results (text)
    with open(os.path.join(output_dir, "low_similarity_results.txt"), "w") as f:
        f.write(f"Results of Filtering Images with Low Similarity to Other Classes\n")
        f.write(f"Total images: {len(image_similarities)}, Selected images: {len(selected_images)}\n")
        f.write("=" * 80 + "\n\n")
        
        for img_path, img_name, max_sim in selected_images:
            f.write(f"{img_name} - Max similarity to other classes: {max_sim:.4f}\n")
    
    # Copy selected images
    print(f"Copying selected images...")
    copied_count = 0
    for _, img_filename, max_sim in tqdm(selected_images, desc="Copying images"):
        # Find source files
        src_files = [f for f in os.listdir(filtered_images_dir) if f.startswith(os.path.splitext(img_filename)[0])]
        
        for src_file in src_files:
            src_path = os.path.join(filtered_images_dir, src_file)
            # New filename (add similarity info)
            name, ext = os.path.splitext(src_file)
            new_name = f"{name}_other_sim{max_sim:.4f}{ext}"
            dst_path = os.path.join(output_images_dir, new_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
    
    print(f"\n=== Filtering Complete ===")
    print(f"Total of {len(filtered_vectors)} images, selected {keep_count} images")
    print(f"{copied_count} images saved to {output_images_dir}")
    
    return selected_images


def calculate_max_similarity(vector: np.ndarray, class_vectors: Dict[str, np.ndarray]) -> float:
    """Calculate maximum similarity between a vector and all vectors in a class"""
    similarities = []
    for class_vec in class_vectors.values():
        similarity = 1 - cosine(vector, class_vec)  # Cosine similarity
        similarities.append(similarity)
    
    return max(similarities) if similarities else 0.0


def filter_keratitis_second_stage():
    """Second stage filtering for keratitis images - select images with low similarity to other classes"""
    print("\n=== Starting Second Stage Filtering for Keratitis Images ===")
    
    # Set paths
    vector_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/vectors"
    filtered_vector_path = os.path.join(vector_dir, "filtered_keratitis.npy")  # First stage filtered vectors
    
    # Set other class vector paths
    other_class_vector_paths = {
        # "normal": os.path.join(vector_dir, "normal.npy"),
        "corneal_ulcer": os.path.join(vector_dir, "각막궤양.npy"),
        # "corneal_sequestrum": os.path.join(vector_dir, "각막부골편.npy"),
        "conjunctivitis": os.path.join(vector_dir, "결막염.npy"),
        # "blepharitis": os.path.join(vector_dir, "blepharitis.npy"),
    }
    
    # Set image paths
    filtered_images_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/matching_filtered/keratitis/filtered_images"
    output_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/matching_filtered/keratitis/second_stage_filtered"
    
    # Run filtering
    try:
        filter_by_class_dissimilarity(
            filtered_vector_path=filtered_vector_path,
            other_class_vector_paths=other_class_vector_paths,
            filtered_images_dir=filtered_images_dir,
            output_dir=output_dir,
            keep_ratio=0.5  # Keep 50%
        )
        print("\n=== Second Stage Filtering Complete ===")
    except Exception as e:
        print(f"Error during second stage filtering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run second stage filtering
    try:
        filter_keratitis_second_stage()
    except Exception as e:
        print(f"Error during keratitis second stage filtering: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Filtering Process Complete ===")
