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
data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/yolo_dataset"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"
test_dir = f"{data_dir}/test"

# 하이퍼파라미터 설정
model_type = "efficientnet-b0"  # EfficientNet 모델 타입
epochs = 100
batch_size = 16
img_size = 224  # EfficientNet에서 요구하는 입력 크기
learning_rate = 0.001
# GPU 설정 (2번과 3번 GPU 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = "/home/minelab/desktop/Jack/step_vet_train/diagnosis/efficient_net/runs"
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
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 데이터 로드
train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])

# 클래스 수와 클래스 이름 가져오기
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print(f"클래스 이름: {class_names}")
print(f"클래스 수: {num_classes}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# EfficientNet 모델 정의
model = EfficientNet.from_pretrained(model_type, num_classes=num_classes)
model = model.to(device)

# GPU가 여러 개 있으면 DataParallel 사용
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 옵티마이저 사용
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 10 epoch마다 학습률 감소

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

# 학습 함수
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, scheduler=None):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        # 각 epoch마다 학습 및 검증 단계
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
                dataloader = train_loader
            else:
                model.eval()  # 모델을 평가 모드로 설정
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반복
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 역전파 + 옵티마이저 단계 (학습 단계에서만)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 통계 업데이트
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            # 손실 및 정확도 기록
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 모델 저장 (검증 정확도가 더 높을 때만)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_efficientnet_model.pth'))
                print(f"Better model found! Saved at epoch {epoch+1}")

        # 학습 단계가 끝난 후 학습률 스케줄러 업데이트
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.6f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        print()

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
    
    print(f"Best val Acc: {best_acc:.4f}")
    
    # 학습 결과 저장
    training_stats = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_acc.item(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_type": model_type,
        "num_classes": num_classes,
        "class_names": class_names
    }
    
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f, indent=4)

# 모델 학습 실행
if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Starting training with {model_type}")
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
    
    # 클래스별 이미지 수 출력
    train_counts = count_images_per_class(train_dataset)
    val_counts = count_images_per_class(val_dataset)
    test_counts = count_images_per_class(test_dataset)
    
    print("클래스별 이미지 수:")
    print("  Train:")
    for cls, count in train_counts.items():
        print(f"    {cls}: {count}")
    print("  Validation:")
    for cls, count in val_counts.items():
        print(f"    {cls}: {count}")
    print("  Test:")
    for cls, count in test_counts.items():
        print(f"    {cls}: {count}")
    
    train_model(model, criterion, optimizer, train_loader, val_loader, epochs, scheduler)
    
    print("Training completed!")
