import os
import shutil
import random
from pathlib import Path

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def select_random_images(source_dir, dest_dir, count=3100):
    # 소스 디렉토리가 존재하는지 확인
    if not os.path.exists(source_dir):
        print(f"Not exist source directory: {source_dir}")
        return
    
    # 목적지 디렉토리 생성
    create_directory(dest_dir)
    
    # 소스 디렉토리에서 모든 이미지 파일 리스트 가져오기
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # 이미지 파일 수가 요청한 수보다 적은 경우 처리
    if len(image_files) < count:
        print(f"경고: {source_dir}에는 {len(image_files)}개의 이미지만 존재하므로 복사 불가능.")
        selected_images = image_files
    else:
        # 무작위로 이미지 선택
        selected_images = random.sample(image_files, count)
    
    # 선택된 이미지 복사
    for image in selected_images:
        src_path = os.path.join(source_dir, image)
        dest_path = os.path.join(dest_dir, image)
        shutil.copy2(src_path, dest_path)
    
    print(f"{source_dir}에서 {len(selected_images)}개 {dest_dir}로 복사 완료.")

def main():
    # 소스 디렉토리 경로
    source_dirs = [
        "/home/minelab/desktop/Jack/step_vet_train/datasets/origin/각막궤양/유",
        "/home/minelab/desktop/Jack/step_vet_train/datasets/origin/각막부골편/유",
        "/home/minelab/desktop/Jack/step_vet_train/datasets/origin/결막염/유"
    ]
    
    # 목적지 디렉토리 경로
    dest_dirs = [
        "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/각막궤양",
        "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/각막부골편",
        "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset/결막염"
    ]
    
    # 각 질병 카테고리별로 이미지 복사
    for src_dir, dst_dir in zip(source_dirs, dest_dirs):
        select_random_images(src_dir, dst_dir, 3100)

if __name__ == "__main__":
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(42)
    main()
