import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# 디렉토리 경로 설정
ORIGIN_DIR = Path("/home/minelab/desktop/Jack/step_vet_train/datasets/origin")
OUTPUT_DIR = Path("/home/minelab/desktop/Jack/step_vet_train/datasets/step1")

# 출력 디렉토리 구조 정의
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

# 각 출력 디렉토리에 normal/abnormal 하위 폴더 생성
for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    (split_dir / "normal").mkdir(parents=True, exist_ok=True)
    (split_dir / "abnormal").mkdir(parents=True, exist_ok=True)

# 임시 디렉토리 (분할 전 이미지 저장)
TEMP_DIR = OUTPUT_DIR / "temp"
TEMP_NORMAL_DIR = TEMP_DIR / "normal"
TEMP_ABNORMAL_DIR = TEMP_DIR / "abnormal"

# 임시 디렉토리 생성
TEMP_NORMAL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_ABNORMAL_DIR.mkdir(parents=True, exist_ok=True)

# 이미지 확장자 목록
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def copy_normal_images(target_count=3100):
    """
    normal 폴더에서 지정된 수의 이미지를 임시 디렉토리로 복사합니다.
    
    Args:
        target_count: 복사할 이미지 수
    
    Returns:
        복사된 이미지 수, 복사된 이미지 경로 리스트
    """
    print(f"정상(normal) 이미지 복사 시작...")
    
    normal_source = ORIGIN_DIR / "normal"
    all_images = []
    
    # normal 폴더에서 이미지 파일 수집
    for img_path in normal_source.glob('**/*'):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            all_images.append(img_path)
    
    print(f"normal 폴더에서 발견된 이미지 수: {len(all_images)}")
    
    # 이미지가 충분하지 않은 경우 경고
    if len(all_images) < target_count:
        print(f"경고: 요청한 {target_count}장보다 적은 {len(all_images)}장만 발견되었습니다.")
    
    # 이미지를 무작위로 섞음
    random.shuffle(all_images)
    
    # 이미 사용된 파일명 추적
    used_filenames = set()
    copied_count = 0
    skipped_count = 0
    copied_images = []
    
    # 목표 개수에 도달할 때까지 이미지 복사
    for img_path in all_images:
        if copied_count >= target_count:
            break
            
        original_filename = img_path.name
        output_path = TEMP_NORMAL_DIR / original_filename
        
        # 파일명 중복 확인
        if original_filename in used_filenames:
            skipped_count += 1
            continue
            
        # 파일명 기록
        used_filenames.add(original_filename)
        
        # 파일 복사
        shutil.copy2(img_path, output_path)
        copied_count += 1
        copied_images.append(output_path)
        
        # 진행 상황 출력 (100개마다)
        if copied_count % 100 == 0:
            print(f"진행 상황: {copied_count}/{target_count} 완료 (건너뛴 파일: {skipped_count}개)")
    
    print(f"정상 이미지 복사 완료: {copied_count}개의 이미지가 임시 디렉토리에 저장되었습니다.")
    return copied_count, copied_images

def copy_abnormal_images(disease_folders, images_per_disease=620):
    """
    지정된 질병 폴더에서 각각 지정된 수의 이미지를 임시 디렉토리로 복사합니다.
    
    Args:
        disease_folders: 질병 폴더 이름 목록
        images_per_disease: 각 질병당 복사할 이미지 수
    
    Returns:
        질병별 복사된 이미지 수 정보, 복사된 이미지 경로 리스트
    """
    print(f"비정상(abnormal) 이미지 복사 시작...")
    
    # 질병별 복사 이미지 수를 추적하는 dictionary
    copied_counts = {}
    copied_images = []
    
    # 이미 사용된 파일명 추적
    used_filenames = set()
    
    # 각 질병 폴더에 대해 처리
    for disease in disease_folders:
        disease_dir = ORIGIN_DIR / disease
        
        if not disease_dir.exists():
            print(f"경고: {disease} 폴더가 존재하지 않습니다.")
            copied_counts[disease] = 0
            continue
        
        print(f"{disease} 폴더에서 이미지 복사 중...")
        
        # 해당 질병 폴더에서 이미지 파일 수집
        all_images = []
        for img_path in disease_dir.glob('**/*'):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                # '무' 폴더에 있는 이미지는 제외 (정상 이미지이므로)
                if not str(img_path).split(os.sep)[-2] == "무":
                    all_images.append(img_path)
        
        print(f"  {disease} 폴더에서 발견된 이미지 수: {len(all_images)}")
        
        # 이미지가 충분하지 않은 경우 경고
        if len(all_images) < images_per_disease:
            print(f"  경고: {disease}에서 요청한 {images_per_disease}장보다 적은 {len(all_images)}장만 발견되었습니다.")
        
        # 이미지를 무작위로 섞음
        random.shuffle(all_images)
        
        # 이 질병에 대해 복사된 이미지 수 추적
        disease_copied = 0
        disease_skipped = 0
        
        # 목표 개수에 도달할 때까지 이미지 복사
        for img_path in all_images:
            if disease_copied >= images_per_disease:
                break
                
            original_filename = img_path.name
            output_path = TEMP_ABNORMAL_DIR / original_filename
            
            # 파일명 중복 확인
            if original_filename in used_filenames:
                disease_skipped += 1
                continue
                
            # 파일명 기록
            used_filenames.add(original_filename)
            
            # 파일 복사
            shutil.copy2(img_path, output_path)
            disease_copied += 1
            copied_images.append(output_path)
            
            # 진행 상황 출력 (100개마다)
            if disease_copied % 100 == 0:
                print(f"  진행 상황: {disease_copied}/{images_per_disease} 완료 (건너뛴 파일: {disease_skipped}개)")
        
        print(f"  {disease} 이미지 복사 완료: {disease_copied}개")
        copied_counts[disease] = disease_copied
    
    total_abnormal = sum(copied_counts.values())
    print(f"비정상 이미지 복사 완료: 총 {total_abnormal}개의 이미지가 임시 디렉토리에 저장되었습니다.")
    
    return copied_counts, copied_images

def split_dataset(normal_images, abnormal_images, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    이미지를 train/val/test로 분할합니다.
    
    Args:
        normal_images: 정상 이미지 경로 리스트
        abnormal_images: 비정상 이미지 경로 리스트
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
    
    Returns:
        각 분할별 이미지 수 정보
    """
    print("\n데이터셋 분할 시작 (train:val:test = 8:1:1)...")
    
    # 정상 이미지 분할
    random.shuffle(normal_images)
    n_normal = len(normal_images)
    n_normal_train = int(n_normal * train_ratio)
    n_normal_val = int(n_normal * val_ratio)
    
    normal_train = normal_images[:n_normal_train]
    normal_val = normal_images[n_normal_train:n_normal_train + n_normal_val]
    normal_test = normal_images[n_normal_train + n_normal_val:]
    
    # 비정상 이미지 분할
    random.shuffle(abnormal_images)
    n_abnormal = len(abnormal_images)
    n_abnormal_train = int(n_abnormal * train_ratio)
    n_abnormal_val = int(n_abnormal * val_ratio)
    
    abnormal_train = abnormal_images[:n_abnormal_train]
    abnormal_val = abnormal_images[n_abnormal_train:n_abnormal_train + n_abnormal_val]
    abnormal_test = abnormal_images[n_abnormal_train + n_abnormal_val:]
    
    # 분할 정보 출력
    print(f"정상(normal) 이미지 분할:")
    print(f"  - 학습(train): {len(normal_train)}장")
    print(f"  - 검증(val): {len(normal_val)}장")
    print(f"  - 테스트(test): {len(normal_test)}장")
    
    print(f"비정상(abnormal) 이미지 분할:")
    print(f"  - 학습(train): {len(abnormal_train)}장")
    print(f"  - 검증(val): {len(abnormal_val)}장")
    print(f"  - 테스트(test): {len(abnormal_test)}장")
    
    # 이미지 복사 작업
    copy_to_split_dirs("normal", normal_train, normal_val, normal_test)
    copy_to_split_dirs("abnormal", abnormal_train, abnormal_val, abnormal_test)
    
    # 임시 디렉토리 정리
    shutil.rmtree(TEMP_DIR)
    print(f"임시 디렉토리 정리 완료")
    
    return {
        "normal": {
            "train": len(normal_train),
            "val": len(normal_val),
            "test": len(normal_test)
        },
        "abnormal": {
            "train": len(abnormal_train),
            "val": len(abnormal_val),
            "test": len(abnormal_test)
        }
    }

def copy_to_split_dirs(class_name, train_images, val_images, test_images):
    """
    분할된 이미지를 최종 디렉토리로 복사합니다.
    
    Args:
        class_name: 클래스 이름 ('normal' 또는 'abnormal')
        train_images: 학습용 이미지 경로 리스트
        val_images: 검증용 이미지 경로 리스트
        test_images: 테스트용 이미지 경로 리스트
    """
    print(f"{class_name} 이미지를 최종 디렉토리로 복사 중...")
    
    # 학습용 이미지 복사
    for img_path in train_images:
        dest_path = TRAIN_DIR / class_name / img_path.name
        shutil.copy2(img_path, dest_path)
    
    # 검증용 이미지 복사
    for img_path in val_images:
        dest_path = VAL_DIR / class_name / img_path.name
        shutil.copy2(img_path, dest_path)
    
    # 테스트용 이미지 복사
    for img_path in test_images:
        dest_path = TEST_DIR / class_name / img_path.name
        shutil.copy2(img_path, dest_path)
    
    print(f"{class_name} 이미지 최종 복사 완료")

def organize_dataset():
    """
    데이터셋을 normal/abnormal로 구분하고 train/val/test로 분할하여 정리합니다.
    """
    print("데이터셋 정리 시작...")
    
    # normal 이미지 복사
    normal_count, normal_images = copy_normal_images(3100)
    
    # abnormal 질병 목록
    disease_folders = ["blepharitis", "keratitis", "각막궤양", "각막부골편", "결막염"]
    
    # abnormal 이미지 복사
    abnormal_counts, abnormal_images = copy_abnormal_images(disease_folders, 620)
    
    # 데이터셋 분할 (train:val:test = 8:1:1)
    split_result = split_dataset(normal_images, abnormal_images)
    
    # 결과 요약
    result = {
        "normal_count": normal_count,
        "abnormal_counts": abnormal_counts,
        "total_abnormal": sum(abnormal_counts.values()),
        "split_result": split_result,
        "train_dir": str(TRAIN_DIR),
        "val_dir": str(VAL_DIR),
        "test_dir": str(TEST_DIR)
    }
    
    return result

if __name__ == "__main__":
    result = organize_dataset()
    print("\n결과 요약:")
    print(f"정상(normal) 이미지: 총 {result['normal_count']}장")
    print(f"비정상(abnormal) 이미지: 총 {result['total_abnormal']}장")
    
    print("\n질병별 이미지 수:")
    for disease, count in result['abnormal_counts'].items():
        print(f"  - {disease}: {count}장")
    
    print("\n데이터셋 분할 결과:")
    print(f"학습(train):")
    print(f"  - 정상(normal): {result['split_result']['normal']['train']}장")
    print(f"  - 비정상(abnormal): {result['split_result']['abnormal']['train']}장")
    
    print(f"검증(val):")
    print(f"  - 정상(normal): {result['split_result']['normal']['val']}장")
    print(f"  - 비정상(abnormal): {result['split_result']['abnormal']['val']}장")
    
    print(f"테스트(test):")
    print(f"  - 정상(normal): {result['split_result']['normal']['test']}장")
    print(f"  - 비정상(abnormal): {result['split_result']['abnormal']['test']}장") 