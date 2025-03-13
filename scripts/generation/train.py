import os
import subprocess

def train_gan(
    outdir="./training-runs/", 
    cfg="", 
    data="", 
    gpus=2, 
    batch=64, 
    mirror=1, 
    snap=50, 
    batch_gpu=8, 
    kimg=2,
    gpu_ids="2, 3"
):
    """
    ProjectedGAN 모델 학습을 위한 함수
    
    매개변수:
        outdir (str): 학습 결과가 저장될 디렉토리
        cfg (str): 모델 설정 
        data (str): 학습 데이터셋 경로
        gpus (int): 사용할 GPU 수
        batch (int): 전체 배치 크기
        mirror (int): 데이터 증강 미러링 여부 (0 또는 1)
        snap (int): 스냅샷 저장 간격 (kimg 단위)
        batch_gpu (int): GPU당 배치 크기
        kimg (int): 학습할 총 이미지 수 (천 단위)
        gpu_ids (str): 사용할 GPU ID (쉼표로 구분)
        
    반환값:
        bool: 학습 성공 여부
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
            f"python /home/minelab/desktop/Jack/projected-gan/train.py --outdir={outdir} --cfg={cfg} --data={data} "
            f"--gpus={gpus} --batch={batch} --mirror={mirror} --snap={snap} "
            f"--batch-gpu={batch_gpu} --kimg={kimg}"
        )
        
        # 명령어 출력 및 실행
        print(f"실행: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"\nGAN 학습 완료!")
        print(f"결과: {outdir}")
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

def main():
    # GAN 학습 매개변수 설정
    outdir = "/home/minelab/desktop/Jack/step_vet_train"
    cfg = "fastgan"
    data = "/home/minelab/desktop/Jack/step_vet_train/datasets/training_gan/keratitis"
    gpus = 2  
    batch = 64 
    mirror = 1
    snap = 50
    batch_gpu = 8  # GPU당 배치 크기
    kimg = 5000
    gpu_ids = "2,3"  # 2번과 3번 GPU 사용
    
    # 출력 디렉토리에 이전 학습 데이터가 있는지 확인
    if os.path.exists(outdir) and any(f.startswith("network-snapshot-") for f in os.listdir(outdir) if os.path.isfile(os.path.join(outdir, f))):
        print(f"경고: 출력 디렉토리 '{outdir}'에 이전 학습 데이터가 존재합니다.")
        choice = input("계속 진행하시겠습니까? (y/n): ").lower()
        if choice != 'y':
            print("GAN 학습을 취소합니다.")
            return False
    
    # GAN 학습 함수 호출
    success = train_gan(
        outdir=outdir,
        cfg=cfg,
        data=data,
        gpus=gpus,
        batch=batch,
        mirror=mirror,
        snap=snap,
        batch_gpu=batch_gpu,
        kimg=kimg,
        gpu_ids=gpu_ids
    )
    
    if success:
        print("GAN 학습이 성공적으로 완료되었습니다.")
    else:
        print("GAN 학습 중 오류가 발생했습니다.")
    
    return success

if __name__ == "__main__":
    main()