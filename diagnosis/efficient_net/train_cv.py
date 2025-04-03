import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# 데이터 경로 설정
data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/yolo_dataset"
train_dir = f"{data_dir}/train"

# 하이퍼파라미터 설정
model_type = "efficientnet-b0"
epochs = 100
batch_size = 16
img_size = 224
learning_rate = 0.001
n_splits = 5  # 5-fold cross validation

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = "/home/minelab/desktop/Jack/step_vet_train/diagnosis/efficient_net/runs_cv"
os.makedirs(output_dir, exist_ok=True)

# 데이터셋 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 데이터 로드
train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])

# 클래스 수와 클래스 이름 가져오기
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print(f"클래스 이름: {class_names}")
print(f"클래스 수: {num_classes}")

# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, epoch, optimizer, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer, path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.val_loss_min = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer, path)

    def save_checkpoint(self, val_loss, model, epoch, optimizer, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)

def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, fold, fold_output_dir):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    for epoch in range(epochs):
        start_time = time.time()
        print(f"Fold {fold}, Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                early_stopping(epoch_loss, model, epoch, optimizer, 
                             os.path.join(fold_output_dir, f'best_model_fold_{fold}.pth'))
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        if early_stopping.early_stop:
            break

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.6f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        print()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 학습 결과 저장
    fold_stats = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_type": model_type,
        "num_classes": num_classes,
        "class_names": class_names
    }
    
    with open(os.path.join(fold_output_dir, f'fold_{fold}_stats.json'), 'w') as f:
        json.dump(fold_stats, f, indent=4)

    return best_acc

def main():
    # K-Fold Cross Validation 설정
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 전체 데이터셋의 인덱스와 seed 이미지 인덱스 분리
    indices = list(range(len(train_dataset)))
    seed_indices = []
    non_seed_indices = []
    
    for idx, (_, label_idx) in enumerate(train_dataset.samples):
        img_path = train_dataset.samples[idx][0]
        if os.path.basename(img_path).startswith('seed'):
            seed_indices.append(idx)
        else:
            non_seed_indices.append(idx)
    
    # 각 fold의 결과를 저장할 리스트
    fold_results = []
    
    # non-seed 이미지에 대해서만 cross validation 수행
    for fold, (train_idx, val_idx) in enumerate(kfold.split(non_seed_indices)):
        print(f"FOLD {fold + 1}/{n_splits}")
        print("-" * 50)
        
        # 현재 fold의 출력 디렉토리 생성
        fold_output_dir = os.path.join(output_dir, f'fold_{fold + 1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # 실제 인덱스로 변환
        train_idx = [non_seed_indices[i] for i in train_idx]
        val_idx = [non_seed_indices[i] for i in val_idx]
        
        # seed 이미지를 학습 세트에 추가
        train_idx.extend(seed_indices)
        
        # 현재 fold의 데이터셋 생성
        train_subsampler = Subset(train_dataset, train_idx)
        val_subsampler = Subset(train_dataset, val_idx)
        
        # DataLoader 생성
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 모델 초기화
        model = EfficientNet.from_pretrained(model_type, num_classes=num_classes)
        model = model.to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        # 손실 함수와 옵티마이저 설정
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # 현재 fold 학습
        best_acc = train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                            fold + 1, fold_output_dir)
        fold_results.append(best_acc)
        
        print(f"Fold {fold + 1} completed with best validation accuracy: {best_acc:.4f}")
        print("-" * 50)
    
    # 전체 결과 저장
    cv_results = {
        "fold_results": fold_results,
        "mean_accuracy": np.mean(fold_results),
        "std_accuracy": np.std(fold_results),
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_type": model_type,
        "num_classes": num_classes,
        "class_names": class_names,
        "seed_images_count": len(seed_indices),
        "non_seed_images_count": len(non_seed_indices)
    }
    
    with open(os.path.join(output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)
    
    print("\nCross Validation Results:")
    print(f"Mean Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in fold_results]}")
    print(f"\nDataset Statistics:")
    print(f"Total images: {len(train_dataset)}")
    print(f"Seed images: {len(seed_indices)}")
    print(f"Non-seed images: {len(non_seed_indices)}")

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Starting {n_splits}-fold cross validation with {model_type}")
    print(f"Train dataset: {len(train_dataset)} images")
    
    main()
    
    print("Cross validation completed!") 