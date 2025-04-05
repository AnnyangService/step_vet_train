import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕 폰트 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 경로 설정
data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset_origin_1400"
val_dir = f"{data_dir}/val"
test_dir = f"{data_dir}/test"

# 하이퍼파라미터 설정
model_type = "efficientnet-b2"  # EfficientNet 모델 타입
batch_size = 32  
img_size = 224  # EfficientNet에서 요구하는 입력 크기
# GPU 설정 (2번과 3번 GPU 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "/home/minelab/desktop/Jack/step_vet_train/models/efficient_net/origin/1400/best_model.pth"
output_dir = "/home/minelab/desktop/Jack/step_vet_train/models/efficient_net/origin/1400/evaluation"
os.makedirs(output_dir, exist_ok=True)

# 데이터셋 전처리
data_transforms = {
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
val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 클래스 이름
class_names = val_dataset.classes
num_classes = len(class_names)
print(f"클래스 이름: {class_names}")
print(f"클래스 수: {num_classes}")

# 모델 로드
def load_model(model_path):
    model = EfficientNet.from_pretrained(model_type, num_classes=num_classes)
    
    # 모델 로드
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model_state_dict']
    
    # DataParallel로 저장된 모델인 경우 처리
    if list(state_dict.keys())[0].startswith('module.'):
        # DataParallel로 저장된 모델
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
    else:
        # 단일 GPU로 저장된 모델
        model.load_state_dict(state_dict)
        # 평가를 위해 DataParallel 사용
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for evaluation!")
            model = nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()
    return model

# 평가 및 결과 저장 함수
def evaluate_and_save_results(model, dataloader, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 메트릭 계산
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # 결과 저장
    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report
    }

    # JSON 저장
    with open(f"{output_dir}/{dataset_name}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 혼동 행렬 시각화 및 저장
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{dataset_name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_confusion_matrix.png")
    plt.close()
    
    # 클래스별 정확도 시각화
    class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum(np.array(all_preds)[class_mask] == i)
            class_acc[class_name] = class_correct / np.sum(class_mask)
        else:
            class_acc[class_name] = 0
    
    plt.figure(figsize=(12, 6))
    plt.bar(class_acc.keys(), class_acc.values())
    plt.title(f'{dataset_name.capitalize()} - 클래스별 정확도')
    plt.xlabel('클래스')
    plt.ylabel('정확도')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_class_accuracy.png")
    plt.close()
    
    # ROC 곡선 (다중 클래스)
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    # One-vs-Rest (OvR) ROC 계산
    plt.figure(figsize=(10, 8))
    all_probs = np.array(all_probs)
    all_labels_bin = np.zeros((len(all_labels), num_classes))
    for i in range(len(all_labels)):
        all_labels_bin[i, all_labels[i]] = 1
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(num_classes), colors):
        if num_classes > 10 and i >= 10:  # 클래스가 너무 많으면 처음 10개만 표시
            break
            
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name.capitalize()} ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_roc_curve.png")
    plt.close()
    
    print(f"{dataset_name.capitalize()} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # 클래스별 정확도 출력
    print("\n  클래스별 정확도:")
    for class_name, accuracy in class_acc.items():
        print(f"    {class_name}: {accuracy:.4f}")
    
    return results

# 오분류 샘플 시각화
def visualize_misclassified(model, dataloader, dataset_name, max_samples=10):
    model.eval()
    misclassified_inputs = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 오분류된 샘플 찾기
            misclassified_mask = (preds != labels)
            if misclassified_mask.sum() > 0:
                misclassified_inputs.extend(inputs[misclassified_mask].cpu())
                misclassified_labels.extend(labels[misclassified_mask].cpu().numpy())
                misclassified_preds.extend(preds[misclassified_mask].cpu().numpy())
                
            if len(misclassified_inputs) >= max_samples:
                break
    
    # 최대 샘플 수만큼 자르기
    misclassified_inputs = misclassified_inputs[:max_samples]
    misclassified_labels = misclassified_labels[:max_samples]
    misclassified_preds = misclassified_preds[:max_samples]
    
    if len(misclassified_inputs) == 0:
        print(f"No misclassified samples found in {dataset_name} dataset.")
        return
    
    # 시각화
    n_rows = (len(misclassified_inputs) + 4) // 5  # 한 행에 최대 5개씩
    fig, axes = plt.subplots(n_rows, min(5, len(misclassified_inputs)), figsize=(15, 3*n_rows))
    if len(misclassified_inputs) <= 5:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    
    for i, (img, true_label, pred_label) in enumerate(zip(misclassified_inputs, misclassified_labels, misclassified_preds)):
        row, col = i // 5, i % 5
        if n_rows > 1 and len(misclassified_inputs) > 5:
            ax = axes[row][col]
        else:
            ax = axes[col]
        
        # 이미지 표시
        img = img.cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_misclassified.png")
    plt.close()

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    # 모델 로드
    model = load_model(model_path)
    
    # 검증 세트 평가
    print("Evaluating on validation set...")
    val_results = evaluate_and_save_results(model, val_loader, "validation")
    visualize_misclassified(model, val_loader, "validation")
    
    # 테스트 세트 평가
    print("Evaluating on test set...")
    test_results = evaluate_and_save_results(model, test_loader, "test")
    visualize_misclassified(model, test_loader, "test")
    
    print(f"Evaluation completed! Results saved to {output_dir}") 