import os
import subprocess
import glob

def generate_images(
    outdir="/home/minelab/desktop/Jack/step_vet_train/datasets/generated/blepharitis",
    network="",
    trunc=1.0,
    seeds="1-10",
    gpu_ids="0"
):
    """
    ProjectedGAN 모델로 이미지를 생성하는 함수
    
    매개변수:
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
        
        print(f"\n이미지 생성 완료!")
        print(f"생성된 이미지: {outdir}")
        
        # 생성된 이미지 개수 확인
        image_count = len(glob.glob(os.path.join(outdir, "*.png")))
        print(f"총 {image_count}개의 이미지가 생성되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

def main():
    model_path = "/home/minelab/desktop/Jack/step_vet_train/models/projected_gan/blepharitis-gpus2-batch64/best_model.pkl"
    
    if model_path:
        print(f"사용할 모델: {model_path}")
    else:
        print("모델 파일을 찾을 수 없습니다.")
        return False
    
    # 이미지 생성 함수 호출
    success = generate_images(
        outdir="/home/minelab/desktop/Jack/step_vet_train/datasets/generated/blepharitis",
        network=model_path,
        trunc=1.0,
        seeds="1-10000",  # 시드 1부터 100까지 총 100개 이미지 생성
        gpu_ids="2,3"  # 사용할 GPU ID
    )
    
    if success:
        print("이미지 생성이 성공적으로 완료되었습니다.")
    else:
        print("이미지 생성 중 오류가 발생했습니다.")
    
    return success

if __name__ == "__main__":
    main()