import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
import seaborn as sns

# 한글 대신 영어로 질병명 매핑
disease_name_map = {
    '각막궤양': 'Corneal Ulcer',
    '각막부골편': 'Corneal Sequestrum',
    '결막염': 'Conjunctivitis',
    '비궤양성각막염': 'Non-ulcerative Keratitis',
    '안검염': 'Blepharitis'
}

# 경로 설정
base_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/origin"

# 데이터를 저장할 딕셔너리 초기화
disease_images = defaultdict(list)  # 질병별 이미지 경로 저장
disease_json_count = Counter()  # 질병별 JSON 파일 수 카운트

# 먼저 데이터 구조 확인 및 카운트
print("데이터셋 구조 확인:")
for disease_dir in os.listdir(base_dir):
    disease_path = os.path.join(base_dir, disease_dir)
    
    if os.path.isdir(disease_path):
        # 영어 질병명으로 변환
        disease_name = disease_name_map.get(disease_dir, disease_dir)
        print(f"\n질병: {disease_name} (원본: {disease_dir})")
        
        # 유/무 하위 폴더 탐색
        for condition_dir in os.listdir(disease_path):
            condition_path = os.path.join(disease_path, condition_dir)
            
            if os.path.isdir(condition_path):
                # JSON 파일 찾기
                json_files = [f for f in os.listdir(condition_path) if f.endswith('.json')]
                jpg_files = [f for f in os.listdir(condition_path) if f.endswith('.jpg')]
                
                print(f"  - {condition_dir} 폴더: {len(json_files)} JSON 파일, {len(jpg_files)} JPG 파일")
                
                # 유 폴더에 있는 파일만 카운트
                if condition_dir == "유":
                    disease_json_count[disease_name] += len(json_files)
                    
                    # 실제 이미지 파일 확인
                    valid_img_count = 0
                    for json_file in json_files:
                        img_file = json_file.replace('.json', '.jpg')
                        img_path = os.path.join(condition_path, img_file)
                        
                        if os.path.exists(img_path):
                            valid_img_count += 1
                            disease_images[disease_name].append(img_path)
                    
                    print(f"    - 실제 매칭되는 이미지 파일: {valid_img_count}개")

# 총 이미지 수 확인
total_images = sum(len(imgs) for imgs in disease_images.values())
print(f"\n총 처리 가능한 이미지 수: {total_images}")

for disease, img_paths in disease_images.items():
    print(f"{disease}: {len(img_paths)}개 이미지")

# 이미지 특성 추출 함수 (향상된 버전)
def extract_image_features(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지를 읽을 수 없음: {img_path}")
            return None
        
        # 이미지 크기 표준화
        img = cv2.resize(img, (256, 256))
        
        # BGR -> HSV 변환
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 히스토그램 계산 (H, S, V 채널)
        hist_h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256])
        
        # 히스토그램 정규화
        hist_h = cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX).flatten()
        hist_s = cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX).flatten()
        hist_v = cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX).flatten()
        
        # 추가 특성: 엣지 히스토그램
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [2], [0, 256]).flatten()
        edge_hist = cv2.normalize(edge_hist, edge_hist, 0, 1, cv2.NORM_MINMAX)
        
        # 모든 특성 결합
        features = np.concatenate([hist_h, hist_s, hist_v, edge_hist])
        return features
    
    except Exception as e:
        print(f"이미지 처리 오류 ({img_path}): {e}")
        return None

# 특성 추출 및 PCA 시각화
disease_features = defaultdict(list)

# 처리 진행상황 표시
print("\n이미지 특성 추출 시작...")
total_to_process = total_images
processed = 0

for disease, img_paths in disease_images.items():
    for img_path in img_paths:
        features = extract_image_features(img_path)
        
        if features is not None:
            disease_features[disease].append(features)
        
        processed += 1
        if processed % 10 == 0:
            print(f"처리 중: {processed}/{total_to_process} 이미지 완료")

# 특성 추출 후 각 질병별 데이터 수 확인
print("\n특성 추출 완료. 질병별 특성 데이터 수:")
for disease, features in disease_features.items():
    print(f"{disease}: {len(features)}개")

# PCA 시각화
all_features = []
disease_labels = []

for disease, features_list in disease_features.items():
    if features_list:  # 데이터가 있는 경우만
        all_features.extend(features_list)
        disease_labels.extend([disease] * len(features_list))

if all_features:
    print(f"\n총 {len(all_features)}개의 특성 데이터로 PCA 수행")
    
    # 데이터 표준화
    all_features_array = np.array(all_features)
    all_features_mean = np.mean(all_features_array, axis=0)
    all_features_std = np.std(all_features_array, axis=0)
    
    # 0으로 나누기 방지
    all_features_std = np.where(all_features_std == 0, 1e-10, all_features_std)
    all_features_normalized = (all_features_array - all_features_mean) / all_features_std
    
    # PCA 적용
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features_normalized)
    
    # 설명된 분산 비율
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA 설명된 분산 비율: PC1={explained_variance[0]:.4f}, PC2={explained_variance[1]:.4f}")
    print(f"총 설명된 분산: {sum(explained_variance):.4f}")
    
    # 시각화
    plt.figure(figsize=(14, 10))
    
    # 질병별 색상 및 마커 설정
    unique_diseases = list(set(disease_labels))
    n_diseases = len(unique_diseases)
    
    # 구분이 잘 되는 색상 팔레트 사용
    color_palette = sns.color_palette("tab10", n_diseases)
    disease_colors = {disease: color_palette[i] for i, disease in enumerate(unique_diseases)}
    
    # 마커 종류 지정
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd']
    disease_markers = {disease: markers[i % len(markers)] for i, disease in enumerate(unique_diseases)}
    
    # 질병별로 산점도 그리기
    for disease in unique_diseases:
        indices = [i for i, d in enumerate(disease_labels) if d == disease]
        x_values = [features_2d[i, 0] for i in indices]
        y_values = [features_2d[i, 1] for i in indices]
        
        plt.scatter(
            x_values, y_values,
            label=f"{disease} (n={len(indices)})",
            color=disease_colors[disease],
            marker=disease_markers[disease],
            s=100,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5
        )
    
    # 각 질병의 중심점 표시
    for disease in unique_diseases:
        indices = [i for i, d in enumerate(disease_labels) if d == disease]
        if indices:  # 데이터가 있는 경우만
            center_x = np.mean([features_2d[i, 0] for i in indices])
            center_y = np.mean([features_2d[i, 1] for i in indices])
            
            # 중심점에 큰 X 표시
            plt.scatter(
                center_x, center_y,
                marker='X',
                s=200,
                color=disease_colors[disease],
                edgecolor='black',
                linewidth=1.5,
                zorder=3
            )
            
            # 약간 오프셋을 준 위치에 질병명 표시
            plt.annotate(
                disease,
                (center_x, center_y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8)
            )
    
    # 그래프 제목 및 레이블 설정
    plt.title('PCA of Disease Image Features', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%})', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%})', fontsize=14)
    
    # 범례 위치 조정 및 크기 설정
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1),
        fontsize=12
    )
    
    # 그리드 추가
    plt.grid(True, alpha=0.3)
    
    # 테두리 추가
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig('disease_pixel_distribution_pca.png', dpi=300, bbox_inches='tight')
    print("PCA 시각화 이미지 저장 완료: disease_pixel_distribution_pca.png")
else:
    print("PCA를 수행할 유효한 특성 데이터가 없습니다.")