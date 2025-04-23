import os
import glob
import numpy as np
from typing import Dict, List, Tuple
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

def vectorize_directories(input_output_pairs: List[Tuple[str, str]], **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
    """
    여러 디렉토리의 이미지를 벡터화하여 각각 저장
    
    Args:
        input_output_pairs (List[Tuple[str, str]]): 각 (입력 디렉토리, 출력 파일 경로) 쌍의 목록
        **kwargs: 설정값들 (checkpoint_path, gpu_id, padding_config, scoring_config)
    
    Returns:
        Dict[str, Dict[str, np.ndarray]]: 각 디렉토리별 벡터화 결과 딕셔너리
    """
    results = {}
    
    for i, (input_dir, output_path) in enumerate(input_output_pairs, 1):
        print(f"\n처리 중 {i}/{len(input_output_pairs)}: {os.path.basename(input_dir)} -> {os.path.basename(output_path)}")
        
        # 출력 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 벡터화 실행
        result = vectorize_directory(
            image_dir=input_dir,
            save_path=output_path,
            **kwargs
        )
        
        results[input_dir] = result
    
    return results

def vectorize_dataset(dataset_dir: str, vectors_dir: str, **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
    """
    데이터셋 디렉토리의 각 클래스 폴더를 자동으로 처리하여 벡터화
    
    Args:
        dataset_dir (str): 데이터셋 디렉토리 (각 클래스별 하위 폴더 포함)
        vectors_dir (str): 벡터 결과를 저장할 디렉토리
        **kwargs: 설정값들
    
    Returns:
        Dict[str, Dict[str, np.ndarray]]: 각 클래스별 벡터화 결과 딕셔너리
    """
    # 디렉토리가 없으면 생성
    os.makedirs(vectors_dir, exist_ok=True)
    
    # 클래스 디렉토리 목록 가져오기
    class_dirs = [d for d in os.listdir(dataset_dir) 
                 if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
    
    input_output_pairs = []
    for class_name in class_dirs:
        input_dir = os.path.join(dataset_dir, class_name)
        output_path = os.path.join(vectors_dir, f"{class_name}.npy")
        input_output_pairs.append((input_dir, output_path))
    
    return vectorize_directories(input_output_pairs, **kwargs)


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
    
    # # 방법 1: 개별 디렉토리 처리
    # input_output_pairs = [
    #     # (입력 디렉토리, 출력 파일 경로)
    #     ("/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/val/결막염", 
    #      "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/vectors/val/결막염.npy"),
    #     ("/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/val/각막부골편", 
    #      "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/vectors/val/각막부골편.npy"),
    #     # 필요에 따라 더 추가
    # ]
    
    # 벡터화 실행 (방법 1)
    # results = vectorize_directories(input_output_pairs, **config)
    
    # 방법 2: 데이터셋 디렉토리 자동 처리
    dataset_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/val"
    vectors_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/vectors/val"
    
    # 벡터화 실행 (방법 2)
    results = vectorize_dataset(dataset_dir, vectors_dir, **config)
    