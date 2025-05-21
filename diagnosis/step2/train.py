import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import confusion_matrix, accuracy_score

# 데이터 경로 설정
data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/step2"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"

# 하이퍼파라미터 설정
model_type = "efficientnet-b2" 
epochs = 100
batch_size = 32 
img_size = 224
learning_rate = 0.0005  
weight_decay = 1e-4  

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = "/home/minelab/desktop/Jack/step_vet_train/models/step2"
os.makedirs(output_dir, exist_ok=True)

# 데이터셋 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  
        transforms.RandomRotation(30),    
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

# 클래스별 이미지 수 계산
class_counts = {}
for _, label_idx in train_dataset.samples:
    class_name = train_dataset.classes[label_idx]
    if class_name in class_counts:
        class_counts[class_name] += 1
    else:
        class_counts[class_name] = 1

# 클래스별 가중치 계산
class_weights = []
for class_name in class_names:
    class_weights.append(1.0 / class_counts[class_name])
class_weights = torch.FloatTensor(class_weights).to(device)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = EfficientNet.from_pretrained(model_type, num_classes=num_classes)
model._dropout = nn.Dropout(0.5) 
model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# 클래스별 이미지 수 계산
def count_images_per_class(dataset):
    counts = {}
    for _, label_idx in dataset.samples:
        label = dataset.classes[label_idx]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts

# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
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
    
    # Early Stopping 초기화
    early_stopping = EarlyStopping(patience=30, verbose=True)
    
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
                # validation accuracy가 best_acc보다 높으면 업데이트
                if epoch_acc > best_acc:
                    best_acc = epoch_acc.item()
                    print(f'New best validation accuracy: {best_acc:.4f}')
                # Learning rate scheduler 업데이트
                if scheduler:
                    scheduler.step(epoch_loss)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

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
        print()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        "model_type": model_type,
        "num_classes": num_classes,
        "class_names": class_names
    }
    
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f, indent=4)

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Starting training with {model_type}")
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
