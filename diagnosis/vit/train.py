import logging
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor, ViTConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from torch.utils.data import DataLoader
from transformers.trainer_callback import EarlyStoppingCallback
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, Resize, ColorJitter, ToTensor
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time

# 불필요한 경고 무시
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

# 로깅 설정
logging.basicConfig(
    filename="debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 데이터셋 경로 설정
data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"

# ViT 모델 및 프로세서 설정
model_name = "google/vit-base-patch16-224"

# 하이퍼파라미터 설정
epochs = 50  # 에폭 수 감소
batch_size = 32
img_size = 224  # ViT 기본 이미지 크기
learning_rate = 5e-5  # 학습률 감소
weight_decay = 1e-4

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = "/home/minelab/desktop/Jack/step_vet_train/models/vit/generated/1400"
os.makedirs(output_dir, exist_ok=True)

# 데이터셋 전처리 - 증강 완화
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 간단한 리사이즈
        transforms.RandomHorizontalFlip(p=0.5),   # 수평 뒤집기만 유지
        transforms.RandomRotation(15),            # 회전 범위 축소
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 색상 변형 감소
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 데이터 로드
train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])

# 클래스 수와 클래스 이름 가져오기
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print(f"클래스 이름: {class_names}")
print(f"클래스 수: {num_classes}")

# id2label 및 label2id 생성
id2label = {i: class_name for i, class_name in enumerate(class_names)}
label2id = {class_name: i for i, class_name in enumerate(class_names)}

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

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ViT 모델 설정
config = ViTConfig.from_pretrained(
    model_name,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.1,  # 드롭아웃 감소
    attention_probs_dropout_prob=0.1  # 드롭아웃 감소
)

# 모델 생성 및 분류기 초기화
model = ViTForImageClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
)

# 구분이 어려운 클래스를 위한 향상된 분류기
# 더 깊은 분류기로 미세한 특징 포착 능력 향상
model.classifier = nn.Sequential(
    nn.BatchNorm1d(model.config.hidden_size),
    nn.Dropout(0.1),
    nn.Linear(model.config.hidden_size, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.1),
    nn.Linear(512, num_classes)
)

# 가중치 초기화
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.classifier.apply(init_weights)

# 모델을 디바이스로 이동
model = model.to(device)

# DataParallel 적용
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# 손실 함수 설정 - 기본 CrossEntropy 사용
criterion = nn.CrossEntropyLoss()

# 레이어별 학습률 설정 (전이학습에 적합하게 조정)
optimizer = optim.AdamW([
    {'params': model.module.vit.embeddings.parameters() if isinstance(model, nn.DataParallel) else model.vit.embeddings.parameters(), 'lr': learning_rate / 10},
    {'params': model.module.vit.encoder.parameters() if isinstance(model, nn.DataParallel) else model.vit.encoder.parameters(), 'lr': learning_rate / 5},
    {'params': model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters(), 'lr': learning_rate}
], weight_decay=weight_decay)

# 스케줄러 설정 - ReduceLROnPlateau로 변경
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

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

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, scheduler=None):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 훈련 시작 시간 기록
    start_time_total = time.time()
    
    # Early Stopping 초기화 - 인내심 증가
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

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # ViT 모델 출력 처리
                    if isinstance(model, nn.DataParallel):
                        outputs = model(inputs).logits
                    else:
                        outputs = model(inputs).logits
                    
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

                running_loss += loss.item() * inputs.size(0)
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
        "model_name": model_name,
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
        'id2label': id2label,
        'label2id': label2id
    }, os.path.join(output_dir, 'final_model.pth'))

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Starting training with {model_name}")
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # 클래스별 이미지 수 출력
    train_counts = count_images_per_class(train_dataset)
    val_counts = count_images_per_class(val_dataset)
    
    print("클래스별 이미지 수:")
    print("  Train:")
    for cls, count in train_counts.items():
        print(f"    {cls}: {count}")
    print("  Validation:")
    for cls, count in val_counts.items():
        print(f"    {cls}: {count}")
    
    train_model(model, criterion, optimizer, train_loader, val_loader, epochs, scheduler)
    
    print("Training completed!")