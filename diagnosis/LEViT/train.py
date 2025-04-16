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

# 경로 추가
import sys
sys.path.append('/home/minelab/desktop/')

# 인코더 관련 모듈 가져오기
from Jack.step_vet_train.query_strategy.encoder.utils.feature_extract import FeatureExtractor
from Jack.step_vet_train.query_strategy.encoder.utils.channel_select import ChannelSelector

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

# 커스텀 데이터셋 생성 (feature vector 사용)
class FeatureVectorDataset(Dataset):
    def __init__(self, image_dir, extractor, selector, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.extractor = extractor
        self.selector = selector
        
        # 클래스 정보 추출
        dataset = datasets.ImageFolder(image_dir)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.samples = dataset.samples
        
        # 결과 저장을 위한 캐시
        self.feature_cache = {}
        
        # 기본 차원 (첫 번째 이미지에서 추출)
        self.vector_dim = 20  # 기본값 설정 (selector.scoring_config에서 top_k)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # 이미 추출한 특징인지 확인
        if image_path in self.feature_cache:
            feature_vector = self.feature_cache[image_path]
        else:
            # 이미지에서 특징 추출
            try:
                features, _ = self.extractor.extract_features(image_path)
                if features is None:
                    # 문제가 있는 이미지는 0으로 채운 벡터 사용
                    print(f"Feature extraction failed for {image_path}")
                    feature_vector = torch.zeros(self.vector_dim)
                else:
                    # 채널 선택 및 벡터화
                    feature_vector = self.selector.get_channel_vector(features)
                    
                    # 차원 동적 업데이트 (첫 성공적인 추출 시)
                    if self.vector_dim == 20 and isinstance(feature_vector, np.ndarray):
                        self.vector_dim = feature_vector.shape[0]
                    
                    feature_vector = torch.from_numpy(feature_vector).float()
                
                # 캐시에 저장
                self.feature_cache[image_path] = feature_vector
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                # 에러 발생 시 0으로 채운 벡터 사용
                feature_vector = torch.zeros(self.vector_dim)
                self.feature_cache[image_path] = feature_vector
        
        return feature_vector, label

# 데이터셋 경로 설정
data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"

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

# 특징 추출기와 채널 선택기 초기화
def initialize_feature_extractors():
    # GPU ID 설정 (CUDA_VISIBLE_DEVICES로 이미 2,3번을 선택했으므로, 0번 인덱스는 실제 2번 GPU)
    print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
    print(f"선택된 GPU 이름: {torch.cuda.get_device_name(0)}")
    
    extractor = FeatureExtractor(
        checkpoint_path=config.get('checkpoint_path'),
        gpu_id=0  # CUDA_VISIBLE_DEVICES="2,3"이므로 0은 실제 시스템의 2번 GPU
    )
    
    selector = ChannelSelector(
        padding_config=config.get('padding_config'),
        scoring_config=config.get('scoring_config')
    )
    
    return extractor, selector

# 데이터 로더 생성
def create_dataloaders(extractor, selector):
    # 멀티프로세싱 워커 수 감소
    num_workers = 1  # CUDA 포크 문제를 방지하기 위해 워커 수 감소
    
    # 커스텀 데이터셋 생성
    train_dataset = FeatureVectorDataset(train_dir, extractor, selector)
    val_dataset = FeatureVectorDataset(val_dir, extractor, selector)
    
    # 클래스 수와 클래스 이름 가져오기
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"클래스 이름: {class_names}")
    print(f"클래스 수: {num_classes}")
    
    # id2label 및 label2id 생성
    id2label = {i: class_name for i, class_name in enumerate(class_names)}
    label2id = {class_name: i for i, class_name in enumerate(class_names)}
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, num_classes, class_names, id2label, label2id

# 클래스별 이미지 수 계산 함수
def count_images_per_class(dataset):
    counts = {}
    for _, label_idx in dataset.samples:
        label = dataset.classes[label_idx]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts

# 모델 생성 함수
def create_model(num_classes, id2label, label2id, feature_dim):
    # 기본 ViT 모델 대신, feature vector를 처리할 수 있는 모델 생성
    class LEViT(nn.Module):
        def __init__(self, feature_dim, num_classes):
            super(LEViT, self).__init__()
            
            # Feature vector를 처리하는 트랜스포머 블록
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=feature_dim, 
                    nhead=8, 
                    dim_feedforward=2048,
                    dropout=0.1, 
                    activation='gelu',
                    batch_first=True
                ), 
                num_layers=6
            )
            
            # 분류기
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            # 입력 형태 변환 (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
            x = x.unsqueeze(1)
            
            # 트랜스포머 통과
            x = self.transformer(x)
            
            # 분류기 통과 (첫 번째 토큰만 사용)
            x = x.squeeze(1)
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
    
    # 특징 추출기 초기화
    print("Initializing feature extractors...")
    extractor, selector = initialize_feature_extractors()
    
    # 데이터 로더 생성
    print("Creating data loaders...")
    train_loader, val_loader, num_classes, class_names, id2label, label2id = create_dataloaders(extractor, selector)
    
    # 특징 벡터 차원 확인 (첫 번째 배치에서 추출)
    for features, _ in train_loader:
        feature_dim = features.shape[1]
        print(f"Feature vector dimension: {feature_dim}")
        break
    
    # 모델 생성
    print("Creating LEViT model...")
    model = create_model(num_classes, id2label, label2id, feature_dim)
    
    # 모델 학습
    print("Starting training...")
    train_model(model, train_loader, val_loader, num_classes, class_names, epochs)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
