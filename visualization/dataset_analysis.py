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
    'blepharitis': 'Blepharitis',
    'keratitis': 'Keratitis',
    '결막염': 'Conjunctivitis',
    '각막궤양': 'Corneal Ulcer',
    '각막부골편': 'Corneal Sequestrum',
    'normal': 'Normal'
}

# 경로 설정
base_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset"

# 데이터를 저장할 딕셔너리 초기화
disease_images = defaultdict(list)  # 질병별 이미지 경로 저장
disease_count = Counter()  # 질병별 이미지 수 카운트

# 데이터셋 구조 확인 및 카운트
print("데이터셋 구조 확인:")
for split in ['train', 'val', 'test']:
    split_path = os.path.join(base_dir, split)
    print(f"\n{split.upper()} 데이터셋:")
    
    for disease_dir in os.listdir(split_path):
        disease_path = os.path.join(split_path, disease_dir)
        
        if os.path.isdir(disease_path):
            # 영어 질병명으로 변환
            disease_name = disease_name_map.get(disease_dir, disease_dir)
            
            # 이미지 파일 찾기
            img_files = [f for f in os.listdir(disease_path) if f.endswith('.jpg')]
            disease_count[disease_name] += len(img_files)
            
            print(f"  - {disease_name}: {len(img_files)}개 이미지")
            
            # 이미지 경로 저장
            for img_file in img_files:
                img_path = os.path.join(disease_path, img_file)
                disease_images[disease_name].append(img_path)

# 총 이미지 수 확인
total_images = sum(len(imgs) for imgs in disease_images.values())
print(f"\n총 이미지 수: {total_images}")

# 데이터 분포 시각화
plt.figure(figsize=(12, 6))
diseases = list(disease_count.keys())
counts = list(disease_count.values())

# 막대 그래프 생성
bars = plt.bar(diseases, counts)

# 막대 위에 수치 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

plt.title('Disease Distribution in Dataset')
plt.xlabel('Disease')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 저장
plt.savefig('disease_distribution.png')
plt.close()

print("\n분포 시각화 이미지 저장 완료: disease_distribution.png")