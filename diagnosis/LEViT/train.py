import logging
import os
import time
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, Resize, ColorJitter, ToTensor
from transformers import ViTForImageClassification, ViTConfig
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch.multiprocessing as mp
import glob
from pathlib import Path

# 경로 추가
import sys
sys.path.append('/home/minelab/desktop/')

# 인코더 관련 모듈 가져오기
from Jack.step_vet_train.query_strategy.encoder.utils.feature_extract import FeatureExtractor
from Jack.step_vet_train.query_strategy.encoder.utils.channel_select import ChannelSelector
from Jack.step_vet_train.query_strategy.encoder.vectorization import vectorize_directory

# 불필요한 경고 무시
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

# 로깅 설정
logging.basicConfig(
    filename="debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 설정
config = {
    'checkpoint_path': "/home/minelab/desktop/Jack/step_vet_train/models/fintuned_sam/best_checkpoint.pth",
    'gpu_id': 3,
    'padding_config': {
        'threshold': 0.7,
        'height_ratio': 10
    },
    'scoring_config': {
        'contrast_weight': 0.5,
        'edge_weight': 0.5,
        'high_percentile': 95,
        'low_percentile': 5,
        'top_k': 20
    }
}

# 멀티프로세싱 시작 방법을 'spawn'으로 설정 - CUDA 포크 오류 해결
mp.set_start_method('spawn', force=True)

# 사전 벡터화된 특징 벡터를 사용하는 데이터셋
class PrecomputedFeatureDataset(Dataset):
    def __init__(self, vectors_dir):
        self.vectors_dir = vectors_dir
        
        # 클래스별 벡터 파일 로드
        self.class_vectors = {}
        self.samples = []
        self.classes = []
        
        # 벡터 디렉토리에서 클래스 이름 가져오기
        for class_file in os.listdir(vectors_dir):
            if class_file.endswith('.npy'):
                class_name = os.path.splitext(class_file)[0]
                self.classes.append(class_name)
                
                # 클래스별 벡터 로드
                vectors_path = os.path.join(vectors_dir, class_file)
                print(f"로드된 벡터 파일: {vectors_path}")
                
                # 벡터 파일 로드 및 형식 확인
                vectors_data = np.load(vectors_path, allow_pickle=True)
                
                # 로드된 데이터가 딕셔너리인 경우 (vectorize_directory의 결과)
                if isinstance(vectors_data, np.ndarray) and vectors_data.dtype == np.dtype('O') and isinstance(vectors_data.item(), dict):
                    # 딕셔너리에서 벡터 추출 (경로 -> 벡터)
                    vectors_dict = vectors_data.item()
                    vectors = np.array(list(vectors_dict.values()))
                    print(f"  딕셔너리 형식 감지: {len(vectors)}개 벡터")
                else:
                    # 기본 배열 형식
                    vectors = vectors_data
                    print(f"  배열 형식 감지: {len(vectors)}개 벡터")
                
                class_idx = len(self.class_vectors)
                self.class_vectors[class_name] = {
                    'vectors': vectors,
                    'idx': class_idx
                }
                
                # 샘플 정보 추가
                for i in range(len(vectors)):
                    self.samples.append((class_name, i, class_idx))
        
        # 클래스 매핑 생성
        self.class_to_idx = {cls: info['idx'] for cls, info in self.class_vectors.items()}
        
        print(f"총 {len(self.samples)}개의 특징 벡터가 로드되었습니다.")
        print(f"클래스: {self.classes}")
        
        if len(self.samples) == 0:
            raise RuntimeError("유효한 특징 벡터가 있는 샘플이 없습니다.")
        
        # 벡터 차원 결정
        first_class = self.classes[0]
        first_vector = self.class_vectors[first_class]['vectors'][0]
        self.vector_dim = first_vector.shape[0]
        print(f"특징 벡터 차원: {self.vector_dim}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        class_name, vector_idx, label = self.samples[idx]
        feature_vector = self.class_vectors[class_name]['vectors'][vector_idx]
        return torch.from_numpy(feature_vector).float(), label

# 데이터셋 경로 설정
data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset"
vectors_dir = f"{data_dir}/vectors"
train_vectors_dir = f"{vectors_dir}/train"
val_vectors_dir = f"{vectors_dir}/val"

# 하이퍼파라미터 설정
epochs = 50
batch_size = 32
learning_rate = 5e-5
weight_decay = 1e-4

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = "/home/minelab/desktop/Jack/step_vet_train/models/levit/"
os.makedirs(output_dir, exist_ok=True)

# 데이터 로더 생성
def create_dataloaders():
    # 사전 계산된 특징 벡터를 사용하는 데이터셋 생성
    train_dataset = PrecomputedFeatureDataset(train_vectors_dir)
    
    # 검증 데이터 처리 - 검증 디렉토리가 있으면 사용, 없으면 훈련 데이터 분할
    if os.path.exists(val_vectors_dir):
        val_dataset = PrecomputedFeatureDataset(val_vectors_dir)
    else:
        print(f"검증 벡터 디렉토리({val_vectors_dir})가 없습니다. 훈련 데이터의 20%를 검증 데이터로 사용합니다.")
        # 훈련 데이터의 80%는 훈련용, 20%는 검증용으로 분할
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # 클래스 수와 클래스 이름 가져오기 (원본 train_dataset에서 가져옴)
    if isinstance(train_dataset, torch.utils.data.Subset):
        # Subset인 경우 dataset 속성으로 원본 데이터셋에 접근
        original_dataset = train_dataset.dataset
        num_classes = len(original_dataset.classes)
        class_names = original_dataset.classes
    else:
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        
    print(f"클래스 이름: {class_names}")
    print(f"클래스 수: {num_classes}")
    
    # id2label 및 label2id 생성
    id2label = {i: class_name for i, class_name in enumerate(class_names)}
    label2id = {class_name: i for i, class_name in enumerate(class_names)}
    
    # feature_dim 가져오기 (원본 train_dataset에서)
    if isinstance(train_dataset, torch.utils.data.Subset):
        feature_dim = train_dataset.dataset.vector_dim
    else:
        feature_dim = train_dataset.vector_dim
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, num_classes, class_names, id2label, label2id, feature_dim

# 클래스별 이미지 수 계산 함수
def count_samples_per_class(dataset):
    counts = {}
    for class_name, _, _ in dataset.samples:
        if class_name in counts:
            counts[class_name] += 1
        else:
            counts[class_name] = 1
    return counts

# 모델 생성 함수
def create_model(num_classes, id2label, label2id, feature_dim):
    # 기본 ViT 모델 대신, feature vector를 처리할 수 있는 모델 생성
    class LEViT(nn.Module):
        def __init__(self, feature_dim, num_classes):
            super(LEViT, self).__init__()
            
            # 토큰화 및 위치 임베딩
            self.token_dim = 256  # 내부 토큰 차원
            self.num_tokens = 8   # 시퀀스 길이
            
            # 특징 벡터를 토큰 시퀀스로 변환
            self.tokenizer = nn.Sequential(
                nn.Linear(feature_dim, self.token_dim * self.num_tokens),
                nn.GELU()
            )
            
            # 위치 임베딩
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.token_dim) * 0.02)
            
            # 트랜스포머 인코더
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.token_dim, 
                    nhead=8, 
                    dim_feedforward=2048,
                    dropout=0.1, 
                    activation='gelu',
                    batch_first=True
                ), 
                num_layers=6
            )
            
            # 분류 토큰 ([CLS] 토큰)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.token_dim) * 0.02)
            
            # 분류기
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.token_dim),
                nn.Linear(self.token_dim, num_classes)
            )
        
        def forward(self, x):
            batch_size = x.shape[0]
            
            # 특징 벡터를 토큰 시퀀스로 변환
            x = self.tokenizer(x)
            x = x.view(batch_size, self.num_tokens, self.token_dim)
            
            # 위치 임베딩 추가
            x = x + self.pos_embedding
            
            # [CLS] 토큰 추가
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # 트랜스포머 통과
            x = self.transformer(x)
            
            # [CLS] 토큰만 사용하여 분류
            x = x[:, 0]
            
            # 분류기 통과
            logits = self.classifier(x)
            
            return logits
    
    # 모델 생성
    model = LEViT(feature_dim, num_classes)
    
    # 가중치 초기화
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # 모델을 디바이스로 이동
    model = model.to(device)
    
    # DataParallel 적용
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    return model

# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        self.val_acc_max = float('-inf')

    def __call__(self, val_acc, model, epoch, optimizer, path):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.save_checkpoint(val_acc, model, epoch, optimizer, path)
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter = 0
            self.val_acc_max = val_acc
            self.save_checkpoint(val_acc, model, epoch, optimizer, path)

    def save_checkpoint(self, val_acc, model, epoch, optimizer, path):
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}). Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, path)

def train_model(model, train_loader, val_loader, num_classes, class_names, epochs):
    # 손실 함수 설정
    criterion = nn.CrossEntropyLoss()
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 스케줄러 설정
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 훈련 시작 시간 기록
    start_time_total = time.time()
    
    # Early Stopping 초기화
    early_stopping = EarlyStopping(patience=15, verbose=True)
    
    # 클래스별 성능 추적을 위한 변수
    all_preds = []
    all_labels = []
    
    # 클래스 간 혼동 분석을 위한 변수
    class_confusion = {}
    for class_name in class_names:
        class_confusion[class_name] = {"correct": 0, "total": 0, 
                                     "confused_with": {other: 0 for other in class_names if other != class_name}}
    
    for epoch in range(epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        # 각 epoch마다 학습 및 검증 단계
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            epoch_preds = []
            epoch_labels = []

            for features, labels in dataloader:
                features = features.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # 모델 출력 처리
                    outputs = model(features)
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # 예측 및 라벨 수집
                if phase == 'val':
                    epoch_preds.extend(preds.cpu().numpy())
                    epoch_labels.extend(labels.cpu().numpy())
                    
                    # 클래스 간 혼동 분석
                    for i in range(len(labels)):
                        true_label = labels[i].item()
                        pred_label = preds[i].item()
                        true_class = class_names[true_label]
                        pred_class = class_names[pred_label]
                        
                        class_confusion[true_class]["total"] += 1
                        if true_label == pred_label:
                            class_confusion[true_class]["correct"] += 1
                        else:
                            class_confusion[true_class]["confused_with"][pred_class] += 1

                running_loss += loss.item() * features.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                
                # 각 클래스별 정확도 계산
                all_preds = epoch_preds
                all_labels = epoch_labels
                
                # 클래스별 정확도 출력
                for i, class_name in enumerate(class_names):
                    class_indices = [j for j, label in enumerate(all_labels) if label == i]
                    if len(class_indices) > 0:
                        class_correct = sum([1 for j in class_indices if all_preds[j] == i])
                        class_acc = class_correct / len(class_indices)
                        print(f"    {class_name} Accuracy: {class_acc:.4f} ({class_correct}/{len(class_indices)})")
                
                # 구분이 어려운 클래스 쌍 분석 - 에폭 마지막에 출력
                if epoch % 5 == 0 or epoch == epochs - 1:
                    print("\n클래스 간 혼동 분석:")
                    for class_name, stats in class_confusion.items():
                        if stats["total"] > 0:
                            acc = stats["correct"] / stats["total"]
                            print(f"  {class_name}: {acc:.4f} ({stats['correct']}/{stats['total']})")
                            
                            # 가장 많이 혼동되는 클래스 찾기
                            if stats["total"] - stats["correct"] > 0:  # 오분류가 있는 경우
                                confused = sorted(stats["confused_with"].items(), 
                                                key=lambda x: x[1], reverse=True)
                                for confused_class, count in confused[:3]:  # 상위 3개만
                                    if count > 0:
                                        confused_pct = count / stats["total"] * 100
                                        print(f"    → {confused_class}로 오분류: {confused_pct:.1f}% ({count}/{stats['total']})")
                    print()
                
                # validation accuracy가 best_acc보다 높으면 업데이트
                if epoch_acc > best_acc:
                    best_acc = epoch_acc.item()
                    print(f'New best validation accuracy: {best_acc:.4f}')
                
                # Learning rate scheduler 업데이트 - 정확도 기반
                if scheduler:
                    scheduler.step(epoch_acc)

            # Early Stopping 체크
            if phase == 'val':
                early_stopping(epoch_acc, model, epoch, optimizer, 
                             os.path.join(output_dir, 'best_model.pth'))
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        if early_stopping.early_stop:
            break
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        
        # 현재 학습률 출력
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']:.8f}")
        
        print()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time_total
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # 학습 과정 시각화 및 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # 혼동 행렬 생성 및 저장
    if len(all_preds) > 0:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 분류 보고서 저장
        cr = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
            json.dump(cr, f, indent=4)
    
    # 학습 결과 저장
    training_stats = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_classes": num_classes,
        "class_names": class_names
    }
    
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f, indent=4)
    
    # 최종 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names,
    }, os.path.join(output_dir, 'final_model.pth'))

def main():
    print(f"Using device: {device}")
    print("Starting LEViT (Lesion Encoder Vision Transformer) training")
    
    # 데이터 로더 생성
    print("Creating data loaders...")
    train_loader, val_loader, num_classes, class_names, id2label, label2id, feature_dim = create_dataloaders()
    
    # 모델 생성
    print("Creating LEViT model...")
    model = create_model(num_classes, id2label, label2id, feature_dim)
    
    # 모델 학습
    print("Starting training...")
    train_model(model, train_loader, val_loader, num_classes, class_names, epochs)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
