import os
import subprocess
import glob

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
    gpu_ids="2,3",
    resume=None
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
        resume (str): 이어서 학습할 체크포인트 파일 경로
        
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
        
        # 이어서 학습할 체크포인트가 있는 경우
        if resume:
            cmd += f" --resume={resume}"
        
        # 명령어 출력 및 실행
        print(f"실행: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"\nGAN 학습 완료!")
        print(f"결과: {outdir}")
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

def find_latest_checkpoint(directory):
    """가장 최신 체크포인트 파일을 찾는 함수"""
    # network-snapshot-*.pkl 형식의 파일 찾기
    checkpoints = glob.glob(os.path.join(directory, "network-snapshot-*.pkl"))
    
    if not checkpoints:
        return None
    
    # 파일 이름에서 숫자 부분 추출하여 정렬
    latest = max(checkpoints, key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))
    return latest

def main():
    # GAN 학습 매개변수 설정
    outdir = "/home/minelab/desktop/Jack/step_vet_train/training_log"
    cfg = "fastgan"
    data = "/home/minelab/desktop/Jack/step_vet_train/datasets/training_gan/keratitis"
    gpus = 2  
    batch = 64 
    mirror = 1
    snap = 50
    batch_gpu = 8
    kimg = 5000
    gpu_ids = "2,3"  
    
    # 특정 체크포인트 파일 직접 지정
    checkpoint_path = "/home/minelab/desktop/Jack/step_vet_train/training_log/keratitis-gpus2-batch64/best_model.pkl"
    
    # 체크포인트 파일이 존재하는지 확인
    if os.path.exists(checkpoint_path):
        print(f"지정된 체크포인트 파일을 사용합니다: {checkpoint_path}")
        resume_path = checkpoint_path
    else:
        print(f"지정된 체크포인트 파일이 존재하지 않습니다. 처음부터 학습을 시작합니다.")
        resume_path = None
    
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
        gpu_ids=gpu_ids,
        resume=resume_path
    )
    
    if success:
        print("GAN 학습이 성공적으로 완료되었습니다.")
    else:
        print("GAN 학습 중 오류가 발생했습니다.")
    
    return success

if __name__ == "__main__":
    main()