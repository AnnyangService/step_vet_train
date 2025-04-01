import os
import subprocess
import glob

def generate_images(
    disease,
    outdir="",
    network="",
    trunc=1.0,
    seeds="1-10",
    gpu_ids="0"
):
    """
    ProjectedGAN 모델로 이미지를 생성하는 함수
    
    매개변수:
        disease (str): 생성할 질병 종류
        outdir (str): 생성된 이미지가 저장될 디렉토리
        network (str): 학습된 모델 파일 경로(.pkl) 또는 URL
        trunc (float): 절단 매개변수(truncation parameter)
        seeds (str): 생성에 사용할 시드 범위 (예: "1-10" 또는 "1,2,3,4,5")
        gpu_ids (str): 사용할 GPU ID (쉼표로 구분)
        
    반환값:
        bool: 이미지 생성 성공 여부
    """
    try:
        # 사용할 GPU 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        
        # 출력 디렉토리가 존재하는지 확인하고 없으면 생성
        if not os.path.exists(outdir):
            print(f"출력 디렉토리 '{outdir}'이(가) 존재하지 않습니다. 생성합니다.")
            os.makedirs(outdir, exist_ok=True)
        
        # 명령어 구성
        cmd = (
            f"python /home/minelab/desktop/Jack/projected-gan/gen_images.py --outdir={outdir} "
            f"--network={network} --trunc={trunc} --seeds={seeds}"
        )
        
        # 명령어 출력 및 실행
        print(f"실행: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"\n{disease} 이미지 생성 완료!")
        print(f"생성된 이미지: {outdir}")
        
        # 생성된 이미지 개수 확인
        image_count = len(glob.glob(os.path.join(outdir, "*.png")))
        print(f"총 {image_count}개의 이미지가 생성되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

def generate_disease(disease):
    """특정 질병에 대한 이미지 생성을 수행하는 함수"""
    print(f"\n{disease} 이미지 생성 시작...")
    
    # 모델 경로 설정
    model_path = f"/home/minelab/desktop/Jack/step_vet_train/models/projected_gan/{disease}/00000-fastgan-{disease}-gpus2-batch64/best_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return False
    
    print(f"사용할 모델: {model_path}")
    
    # 이미지 생성 함수 호출
    success = generate_images(
        disease=disease,
        outdir=f"/home/minelab/desktop/Jack/step_vet_train/datasets/generated/{disease}",
        network=model_path,
        trunc=1.0,
        seeds="1-10000",  # 시드 1부터 10000까지 총 10000개 이미지 생성
        gpu_ids="2,3"  # 사용할 GPU ID
    )
    
    if success:
        print(f"{disease} 이미지 생성이 성공적으로 완료되었습니다.")
    else:
        print(f"{disease} 이미지 생성 중 오류가 발생했습니다.")
    
    return success

def main():
    diseases = ['blepharitis', 'keratitis']
    
    for disease in diseases:
        success = generate_disease(disease)
        if not success:
            print(f"{disease} 이미지 생성 중 오류가 발생하여 프로그램을 종료합니다.")
            return False
    
    print("\n모든 질병에 대한 이미지 생성이 완료되었습니다!")
    return True

if __name__ == "__main__":
    main() 