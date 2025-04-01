import os
import subprocess

def convert_dataset(source, dest, resolution=256, gpu_ids="2,3"):
    """
    ProjectedGAN 학습용 데이터셋을 변환하는 함수
    
    매개변수:
        source (str): 원본 데이터셋 경로
        dest (str): 변환된 데이터셋 저장 경로 (.zip)
        resolution (int): 이미지 해상도 (기본값: 256)
        gpu_ids (str): 사용할 GPU ID (쉼표로 구분, 기본값: "2,3")
        
    반환값:
        bool: 변환 성공 여부
    """
    try:
        # 사용할 GPU 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        
        # 명령어 구성
        cmd = f"python /home/minelab/desktop/Jack/projected-gan/dataset_tool.py --source={source} --dest={dest} --resolution={resolution}x{resolution} --transform=center-crop"
        
        # 명령어 출력 및 실행
        print(f"실행: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"\n데이터셋 변환 완료!")
        print(f"결과: {dest}")
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

def prepare_disease_dataset(korean_name, english_name):
    """특정 질병에 대한 데이터셋을 준비하는 함수"""
    print(f"\n{korean_name} ({english_name}) 데이터셋 준비 시작...")
    
    # 데이터셋 변환 매개변수 설정
    source = f"/home/minelab/desktop/Jack/step_vet_train/datasets/origin/{korean_name}/유"
    dest = f"/home/minelab/desktop/Jack/step_vet_train/datasets/training_gan/{english_name}"
    resolution = 256
    gpu_ids = "2,3"
    
    # 이미 대상 파일이 존재하는지 확인
    if os.path.exists(dest):
        print(f"경고: 대상 파일 '{dest}'이(가) 이미 존재합니다.")
        print("데이터셋 변환을 건너뜁니다.")
        return True
    
    # 대상 디렉토리가 존재하는지 확인하고 없으면 생성
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        print(f"대상 디렉토리 '{dest_dir}'이(가) 존재하지 않습니다. 생성합니다.")
        os.makedirs(dest_dir, exist_ok=True)
    
    # 데이터셋 변환 함수 호출
    success = convert_dataset(
        source=source,
        dest=dest,
        resolution=resolution,
        gpu_ids=gpu_ids
    )
    
    if success:
        print(f"{korean_name} ({english_name}) 데이터셋 변환 완료")
    else:
        print(f"{korean_name} ({english_name}) 데이터셋 변환 실패")
    
    return success

def main():
    # 두 질병 모두에 대해 데이터셋 준비
    disease_mapping = {
        '안검염': 'blepharitis',
        '비궤양성각막염': 'keratitis'
    }
    
    for korean_name, english_name in disease_mapping.items():
        success = prepare_disease_dataset(korean_name, english_name)
        if not success:
            print(f"{korean_name} ({english_name}) 데이터셋 준비 중 오류가 발생하여 프로그램을 종료합니다.")
            return False
    
    print("\n모든 질병에 대한 데이터셋 준비가 완료되었습니다!")
    return True

if __name__ == "__main__":
    main()