from Jack.step_vet_train.finetuning_sam.utils.dataloader import load_dataset
from Jack.step_vet_train.finetuning_sam.utils.sam_dataset import SAMDataset
from Jack.step_vet_train.finetuning_sam.utils.mask_processor import BlueMaskExtractor, MaskProcessor
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
import os
import gc
import json
from pathlib import Path
import argparse

class SAMFineTuner:
    """
    Segment Anything Model 파인튜닝 클래스
    """
    def __init__(self, 
                 image_dir: str,
                 mask_dir: str,
                 save_dir: str,
                 model_name: str = "facebook/sam-vit-base",
                 batch_size: int = 4,
                 num_epochs: int = 50,
                 learning_rate: float = 1e-6,
                 weight_decay: float = 0,
                 gpu_ids: str = "0, 1"):
        """
        Args:
            image_dir: 이미지 디렉토리 경로
            mask_dir: 마스크 디렉토리 경로
            save_dir: 모델 저장 디렉토리 경로
            model_name: 사용할 SAM 모델 이름
            batch_size: 배치 사이즈
            num_epochs: 학습 에폭 수
            learning_rate: 학습률
            weight_decay: 가중치 감쇠
            gpu_ids: 사용할 GPU ID 문자열
        """
        # 설정 저장
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.save_dir = save_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # GPU 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
        gc.collect()
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 학습 관련 변수 초기화
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # 학습 히스토리
        self.history = {
            'losses': [],
            'learning_rates': []
        }
        
        # 데이터, 모델, 옵티마이저 초기화
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
    def setup_data(self):
        """데이터셋과 데이터로더 설정"""
        # 데이터셋 로드
        dataset = load_dataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir
        )
        
        # SAM 프로세서와 데이터셋 생성
        self.processor = SamProcessor.from_pretrained(self.model_name)
        self.train_dataset = SAMDataset(dataset=dataset, processor=self.processor)
        
        # 데이터로더 생성
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def setup_model(self):
        """모델 설정 및 로드"""
        # 모델 로드
        self.model = SamModel.from_pretrained(self.model_name)
        
        # vision_encoder만 학습 가능하도록 설정
        for name, param in self.model.named_parameters():
            if name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
                param.requires_grad_(False)
            else:  # vision_encoder
                param.requires_grad_(True)
                
        # 다중 GPU 설정
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        
        # 이전 체크포인트 불러오기
        self.load_last_checkpoint()
    
    def setup_optimizer(self):
        """옵티마이저 및 손실 함수 설정"""
        self.optimizer = Adam(
            self.model.vision_encoder.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        self.seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, 
            squared_pred=True, 
            reduction='mean'
        )
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'vision_encoder_state_dict': {
                name.replace('module.', ''): param 
                for name, param in self.model.state_dict().items() 
                if 'vision_encoder' in name
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        # 일반 체크포인트 저장
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
    
    def load_last_checkpoint(self):
        """마지막 체크포인트 불러오기"""
        checkpoint_files = [f for f in os.listdir(self.save_dir) 
                          if f.startswith('checkpoint_epoch_')]
        
        if checkpoint_files:
            # 가장 마지막 checkpoint 찾기
            last_checkpoint = max(checkpoint_files, 
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(self.save_dir, last_checkpoint)
            
            checkpoint = torch.load(checkpoint_path)
            
            # 모델 가중치 로드
            current_state_dict = self.model.state_dict()
            for name, param in checkpoint['vision_encoder_state_dict'].items():
                current_state_dict[f'module.{name}'] = param
            self.model.load_state_dict(current_state_dict)
            
            # 옵티마이저 로드
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 에폭과 loss 설정
            self.start_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['loss']
            
            print(f"Resuming from epoch {self.start_epoch} with best loss: {self.best_loss}")
    
    def save_history(self):
        """학습 히스토리 저장"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
    
    def train(self):
        """모델 학습 실행"""
        self.model.train()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f'\nCurrent Epoch: {epoch}/{self.num_epochs-1}')
            epoch_losses = []
            
            for batch in tqdm(self.train_dataloader):
                # 모델 출력
                outputs = self.model(
                    pixel_values=batch["pixel_values"].to(self.device),
                    input_boxes=batch["input_boxes"].to(self.device),
                    multimask_output=False
                )
                
                # 손실 계산
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
                loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                
                # 역전파 및 최적화
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # 평균 손실 계산
            mean_loss = mean(epoch_losses)
            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean_loss}')
            
            # 히스토리 업데이트
            self.history['losses'].append(mean_loss)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 체크포인트 저장
            is_best = mean_loss < self.best_loss
            if is_best:
                self.best_loss = mean_loss
                
            self.save_checkpoint(epoch + 1, mean_loss, is_best)
            
            # 히스토리 저장
            self.save_history()
        
        print("Training finished!")

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 모델 파인튜닝")
    
    parser.add_argument("--image_dir", type=str, 
                      default="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases",
                      help="학습 이미지 디렉토리 경로")
    
    parser.add_argument("--mask_dir", type=str,
                      default="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks/tiff_masks",
                      help="마스크 이미지 디렉토리 경로")
    
    parser.add_argument("--save_dir", type=str,
                      default="/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models",
                      help="모델 저장 디렉토리 경로")
    
    parser.add_argument("--model", type=str,
                      default="facebook/sam-vit-base",
                      help="사용할 SAM 모델 이름")
    
    parser.add_argument("--batch_size", type=int, default=4,
                      help="배치 사이즈")
    
    parser.add_argument("--epochs", type=int, default=50,
                      help="학습 에폭 수")
    
    parser.add_argument("--lr", type=float, default=1e-6,
                      help="학습률")
    
    parser.add_argument("--weight_decay", type=float, default=0,
                      help="가중치 감쇠")
    
    parser.add_argument("--gpu_ids", type=str, default="0, 1",
                      help="사용할 GPU ID (예: '0,1')")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # SAM 파인튜너 생성 및 학습 실행
    finetuner = SAMFineTuner(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        save_dir=args.save_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gpu_ids=args.gpu_ids
    )
    
    # 학습 실행
    finetuner.train()