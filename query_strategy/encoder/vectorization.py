import os
import glob
import numpy as np
from typing import Dict
import sys
import torch

# 경로 추가
sys.path.append('/home/minelab/desktop/')
from Jack.step_vet_train.query_strategy.encoder.utils.feature_extract import FeatureExtractor
from Jack.step_vet_train.query_strategy.encoder.utils.channel_select import ChannelSelector

def vectorize_directory(image_dir: str, save_path: str, **kwargs) -> Dict[str, np.ndarray]:
    """
    디렉토리 내의 모든 이미지를 벡터화하여 저장
    
    Args:
        image_dir (str): 이미지가 있는 디렉토리 경로
        save_path (str): 결과를 저장할 경로
        **kwargs: 설정값들 (checkpoint_path, gpu_id, padding_config, scoring_config)
    
    Returns:
        Dict[str, np.ndarray]: 이미지 경로를 키로, 특징 벡터를 값으로 하는 딕셔너리
    """
    # 특징 추출기와 채널 선택기 초기화
    extractor = FeatureExtractor(
        checkpoint_path=kwargs.get('checkpoint_path'),
        gpu_id=kwargs.get('gpu_id', 3)
    )
    
    selector = ChannelSelector(
        padding_config=kwargs.get('padding_config'),
        scoring_config=kwargs.get('scoring_config')
    )
    
    # 지원하는 이미지 확장자
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    results = {}
    
    # 모든 이미지 파일 찾기
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    # 각 이미지 처리
    print(f"Found {len(image_files)} images to process")
    for i, image_path in enumerate(image_files, 1):
        try:
            # 진행상황 출력 (같은 줄에 업데이트)
            print(f"\rProcessing image {i}/{len(image_files)}: {os.path.basename(image_path)}", 
                  end='', flush=True)
            
            # 1. 특징 추출
            features, _ = extractor.extract_features(image_path)
            if features is None:
                print(f"\nSkipping {image_path}: Feature extraction failed")
                continue
                
            # 2. 채널 선택 및 벡터화
            feature_vector = selector.get_channel_vector(features)
            
            # 상대 경로로 저장 (키로 사용)
            rel_path = os.path.relpath(image_path, image_dir)
            results[rel_path] = feature_vector
            
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")

    print("\nVectorization completed!")
    
    # 결과 저장
    np.save(save_path, results)
    print(f"Results saved to {save_path}")
    
    return results


if __name__ == "__main__":
    # 설정
    config = {
        'checkpoint_path': "/home/minelab/desktop/Jack/step_vet_train/models/fintuned_sam/best_checkpoint.pth",
        'gpu_id': 3,
        'padding_config': {
            'threshold': 0.7,
            'height_ratio': 10
        },
        'scoring_config': {
            'contrast_weight': 0.5,
            'edge_weight': 0.5,
            'high_percentile': 95,
            'low_percentile': 5,
            'top_k': 20
        }
    }
    
    # 입력 디렉토리와 결과 저장 경로 설정
    input_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/generated/keratitis"
    output_path = "/home/minelab/desktop/Jack/step_vet_train/datasets/vectors/generated_keratitis.npy"
    
    # 벡터화 실행
    feature_vectors = vectorize_directory(
        image_dir=input_dir,
        save_path=output_path,
        **config
    )