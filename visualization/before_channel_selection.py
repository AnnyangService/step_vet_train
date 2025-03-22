import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import sys

from Jack.step_vet_train.query_strategy.encoder.encoder import CustomSamEncoder

def visualize_features(features: torch.Tensor, original_image: Image.Image, save_dir: str):
    """
    특징맵을 시각화하고 저장합니다.
    
    Args:
        features (torch.Tensor): 인코더에서 추출한 특징맵
        original_image (PIL.Image): 원본 이미지
        save_dir (str): 결과를 저장할 디렉토리 경로
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 원본 이미지와 평균 feature map
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    feature_mean = features[0].mean(dim=0)
    plt.imshow(feature_mean.numpy(), cmap='viridis')
    plt.title('Average Feature Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_map_avg.png'))
    plt.close()
    
    # 2. 개별 채널 시각화 (처음 64개)
    num_channels = 64
    rows = 8
    cols = 8
    plt.figure(figsize=(20, 20))
    
    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(features[0, i].numpy(), cmap='viridis')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_map_channels.png'))
    plt.close()

if __name__ == "__main__":
    # 설정
    checkpoint_path = "/home/minelab/desktop/Jack/step_vet_train/models/fintuned_sam/best_checkpoint.pth"
    save_dir = "/home/minelab/desktop/Jack/step_vet_train/visualization/results/encoder/before_channel_selection"
    test_image = "/home/minelab/desktop/Jack/step_vet_train/visualization/results/encoder/origin.jpg"
    
    # 특징 추출
    encoder = CustomSamEncoder(checkpoint_path=checkpoint_path, gpu_id=3)
    features, original_image = encoder.extract_features(test_image)
    
    # 시각화
    visualize_features(features, original_image, save_dir)