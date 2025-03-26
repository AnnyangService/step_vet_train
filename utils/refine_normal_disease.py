import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# 원본 데이터셋 경로와 출력 경로 설정
ORIGIN_DIR = Path("/home/minelab/desktop/Jack/step_vet_train/datasets/origin")
OUTPUT_DIR = Path("/home/minelab/desktop/Jack/step_vet_train/datasets/refined_matching/refined_dataset/normal")

# 출력 디렉토리가 없으면 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def collect_negative_images(target_count=3100):
    """
    한글 질병 폴더 내의 '무' 폴더에서 이미지 수집
    
    Args:
        target_count: 수집할 총 이미지 수
    """
    # 이미지 확장자 목록
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 모든 이미지 파일 경로를 저장할 리스트
    all_images = []
    
    # 디버깅을 위한 폴더별 이미지 수 카운트
    folder_counts = defaultdict(int)
    
    # 모든 질병 폴더 순회
    for disease_folder in ORIGIN_DIR.iterdir():
        if disease_folder.is_dir():
            # 각 질병 폴더 내에서 '무' 폴더 찾기
            for subfolder in disease_folder.iterdir():
                if subfolder.is_dir() and subfolder.name == "무":
                    print(f"'무' 폴더 발견: {subfolder}")
                    
                    # 각 '무' 폴더에서 이미지 파일 수집
                    for img_path in subfolder.glob('**/*'):
                        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                            all_images.append(img_path)
                            folder_counts[str(disease_folder.name)] += 1
    
    # 총 발견된 이미지 수
    print(f"총 발견된 이미지 수: {len(all_images)}")
    for folder, count in folder_counts.items():
        print(f"  - {folder}: {count}장")
    
    # 충분한 이미지가 없는 경우 경고
    if len(all_images) < target_count:
        print(f"경고: 요청한 {target_count}장보다 적은 {len(all_images)}장만 발견되었습니다.")
    
    # 이미 사용된 파일명 추적
    used_filenames = set()
    copied_count = 0
    skipped_count = 0
    
    # 모든 이미지를 무작위로 섞음
    random.shuffle(all_images)
    
    # 목표 개수에 도달할 때까지 이미지 복사
    for img_path in all_images:
        if copied_count >= target_count:
            break
            
        original_filename = img_path.name
        output_path = OUTPUT_DIR / original_filename
        
        # 파일명 중복 확인
        if original_filename in used_filenames:
            skipped_count += 1
            continue
            
        # 파일명 기록
        used_filenames.add(original_filename)
        
        # 파일 복사
        shutil.copy2(img_path, output_path)
        copied_count += 1
        
        # 진행 상황 출력 (100개마다)
        if copied_count % 100 == 0:
            print(f"진행 상황: {copied_count}/{target_count} 완료 (건너뛴 파일: {skipped_count}개)")
    
    print(f"\n작업 완료: {copied_count}개의 이미지가 {OUTPUT_DIR}에 저장되었습니다.")
    print(f"건너뛴 중복 파일: {skipped_count}개")
    
    # 목표 개수에 도달하지 못한 경우 경고
    if copied_count < target_count:
        print(f"경고: 중복을 제외하고 {copied_count}개의 고유 이미지만 찾았습니다. 목표인 {target_count}개에 도달하지 못했습니다.")
    
    # 결과 요약 반환
    return {
        "total_found": len(all_images),
        "total_copied": copied_count,
        "total_skipped": skipped_count,
        "output_dir": str(OUTPUT_DIR)
    }

if __name__ == "__main__":
    result = collect_negative_images(3100)
    print(result) 