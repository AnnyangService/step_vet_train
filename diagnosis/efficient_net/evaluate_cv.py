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
import glob

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕 폰트 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 경로 설정
data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/dataset"
val_dir = f"{data_dir}/val"
test_dir = f"{data_dir}/test"

# 하이퍼파라미터 설정
model_type = "efficientnet-b2"  # EfficientNet 모델 타입
batch_size = 32  
img_size = 224  # EfficientNet에서 사용하는 입력 크기

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 경로 설정
model_base_dir = "/home/minelab/desktop/Jack/step_vet_train/models/efficient_net/generated_cv/1400"
output_dir = f"{model_base_dir}/evaluation"
os.makedirs(output_dir, exist_ok=True)

# K-fold 수 설정
n_splits = 5

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

# 모델 로드 함수
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

# 단일 모델 평가 함수
def evaluate_single_model(model, dataloader, dataset_name, fold=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # EfficientNet 모델 출력 처리
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

    # 폴드 정보를 파일 이름에 추가
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    result_filename = f"{dataset_name}{fold_suffix}_results.json"
    
    # JSON 저장
    with open(os.path.join(output_dir, result_filename), "w") as f:
        json.dump(results, f, indent=4)

    # 혼동 행렬 시각화 및 저장
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    fold_title = f"Fold {fold} - " if fold is not None else ""
    plt.title(f"{fold_title}{dataset_name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}{fold_suffix}_confusion_matrix.png"))
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
    plt.title(f'{fold_title}{dataset_name.capitalize()} - 클래스별 정확도')
    plt.xlabel('클래스')
    plt.ylabel('정확도')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}{fold_suffix}_class_accuracy.png"))
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
    plt.title(f'{fold_title}{dataset_name.capitalize()} ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}{fold_suffix}_roc_curve.png"))
    plt.close()
    
    fold_text = f"Fold {fold} " if fold is not None else ""
    print(f"{fold_text}{dataset_name.capitalize()} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # 클래스별 정확도 출력
    print("\n  클래스별 정확도:")
    for class_name, accuracy in class_acc.items():
        print(f"    {class_name}: {accuracy:.4f}")
    
    return results, all_probs

# 앙상블 평가 함수
def evaluate_ensemble(models, dataloader, dataset_name):
    all_labels = []
    all_ensemble_preds = []
    all_ensemble_probs = []

    # 데이터 배치별로 각 모델의 예측 수집
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            batch_probs = []
            
            # 각 모델에서 예측 확률 수집
            for model in models:
                model.eval()
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            # 앙상블 예측 확률 (평균)
            ensemble_probs = np.mean(batch_probs, axis=0)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
            all_ensemble_probs.extend(ensemble_probs)
            all_ensemble_preds.extend(ensemble_preds)
            all_labels.extend(labels.cpu().numpy())

    # 메트릭 계산
    acc = accuracy_score(all_labels, all_ensemble_preds)
    precision = precision_score(all_labels, all_ensemble_preds, average='weighted')
    recall = recall_score(all_labels, all_ensemble_preds, average='weighted')
    f1 = f1_score(all_labels, all_ensemble_preds, average='weighted')
    
    conf_matrix = confusion_matrix(all_labels, all_ensemble_preds)
    class_report = classification_report(all_labels, all_ensemble_preds, target_names=class_names, output_dict=True)
    
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
    with open(os.path.join(output_dir, f"{dataset_name}_ensemble_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # 혼동 행렬 시각화 및 저장
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Ensemble - {dataset_name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_ensemble_confusion_matrix.png"))
    plt.close()
    
    # 클래스별 정확도 시각화
    class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum(np.array(all_ensemble_preds)[class_mask] == i)
            class_acc[class_name] = class_correct / np.sum(class_mask)
        else:
            class_acc[class_name] = 0
    
    plt.figure(figsize=(12, 6))
    plt.bar(class_acc.keys(), class_acc.values())
    plt.title(f'Ensemble - {dataset_name.capitalize()} - 클래스별 정확도')
    plt.xlabel('클래스')
    plt.ylabel('정확도')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_ensemble_class_accuracy.png"))
    plt.close()
    
    # ROC 곡선 (다중 클래스)
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    # One-vs-Rest (OvR) ROC 계산
    plt.figure(figsize=(10, 8))
    all_ensemble_probs = np.array(all_ensemble_probs)
    all_labels_bin = np.zeros((len(all_labels), num_classes))
    for i in range(len(all_labels)):
        all_labels_bin[i, all_labels[i]] = 1
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(num_classes), colors):
        if num_classes > 10 and i >= 10:  # 클래스가 너무 많으면 처음 10개만 표시
            break
            
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_ensemble_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Ensemble - {dataset_name.capitalize()} ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_ensemble_roc_curve.png"))
    plt.close()
    
    print(f"Ensemble {dataset_name.capitalize()} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # 클래스별 정확도 출력
    print("\n  클래스별 정확도:")
    for class_name, accuracy in class_acc.items():
        print(f"    {class_name}: {accuracy:.4f}")
    
    return results

# 오분류 샘플 시각화 함수
def visualize_ensemble_misclassified(models, dataloader, dataset_name, max_samples=10):
    all_labels = []
    all_ensemble_preds = []
    all_images = []
    
    # 데이터 배치별로 각 모델의 예측 수집
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            batch_probs = []
            
            # 각 모델에서 예측 확률 수집
            for model in models:
                model.eval()
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            # 앙상블 예측 (평균)
            ensemble_probs = np.mean(batch_probs, axis=0)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
            # 오분류 찾기
            misclassified_mask = (ensemble_preds != labels.cpu().numpy())
            
            if np.any(misclassified_mask):
                for idx, is_miss in enumerate(misclassified_mask):
                    if is_miss:
                        all_images.append(inputs[idx].cpu())
                        all_labels.append(labels[idx].cpu().item())
                        all_ensemble_preds.append(ensemble_preds[idx])
                        
                        if len(all_images) >= max_samples:
                            break
                
            if len(all_images) >= max_samples:
                break
    
    if len(all_images) == 0:
        print(f"No misclassified samples found in {dataset_name} dataset.")
        return
    
    # 시각화
    n_rows = (len(all_images) + 4) // 5  # 한 행에 최대 5개씩
    fig, axes = plt.subplots(n_rows, min(5, len(all_images)), figsize=(15, 3*n_rows))
    if len(all_images) <= 5:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    
    for i, (img, true_label, pred_label) in enumerate(zip(all_images, all_labels, all_ensemble_preds)):
        row, col = i // 5, i % 5
        if n_rows > 1 and len(all_images) > 5:
            ax = axes[row][col]
        else:
            ax = axes[col]
        
        # 이미지 표시
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_ensemble_misclassified.png"))
    plt.close()

def main():
    # 폴드별 결과 저장
    fold_results_val = []
    fold_results_test = []
    
    # 모든 폴드 모델 로드
    loaded_models = []
    
    for fold in range(1, n_splits + 1):
        model_path = os.path.join(model_base_dir, f"fold_{fold}", f"best_model_fold_{fold}.pth")
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = load_model(model_path)
            loaded_models.append(model)
            
            # 각 폴드 모델 평가
            print(f"\nEvaluating Fold {fold} model on validation set...")
            val_result, _ = evaluate_single_model(model, val_loader, "validation", fold)
            fold_results_val.append(val_result)
            
            print(f"\nEvaluating Fold {fold} model on test set...")
            test_result, _ = evaluate_single_model(model, test_loader, "test", fold)
            fold_results_test.append(test_result)
        else:
            print(f"Model not found at path: {model_path}")
    
    # 폴드 평균 성능 계산
    if len(fold_results_val) > 0:
        val_acc_mean = np.mean([r["accuracy"] for r in fold_results_val])
        val_acc_std = np.std([r["accuracy"] for r in fold_results_val])
        test_acc_mean = np.mean([r["accuracy"] for r in fold_results_test])
        test_acc_std = np.std([r["accuracy"] for r in fold_results_test])
        
        print("\nFold Models Average Performance:")
        print(f"Validation Accuracy: {val_acc_mean:.4f} ± {val_acc_std:.4f}")
        print(f"Test Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    
    # 앙상블 평가
    if len(loaded_models) > 1:
        print("\nEvaluating Ensemble model on validation set...")
        val_ensemble_result = evaluate_ensemble(loaded_models, val_loader, "validation")
        
        print("\nEvaluating Ensemble model on test set...")
        test_ensemble_result = evaluate_ensemble(loaded_models, test_loader, "test")
        
        # 앙상블 오분류 시각화
        print("\nVisualizing ensemble misclassified examples...")
        visualize_ensemble_misclassified(loaded_models, test_loader, "test")
        
        # 성능 비교: 개별 폴드 vs 앙상블
        print("\nPerformance Comparison:")
        print(f"Average Fold Validation Accuracy: {val_acc_mean:.4f} ± {val_acc_std:.4f}")
        print(f"Ensemble Validation Accuracy: {val_ensemble_result['accuracy']:.4f}")
        print(f"Average Fold Test Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}")
        print(f"Ensemble Test Accuracy: {test_ensemble_result['accuracy']:.4f}")
        
        # 결과 저장
        comparison_results = {
            "validation": {
                "fold_mean_accuracy": val_acc_mean,
                "fold_std_accuracy": val_acc_std,
                "ensemble_accuracy": val_ensemble_result["accuracy"]
            },
            "test": {
                "fold_mean_accuracy": test_acc_mean,
                "fold_std_accuracy": test_acc_std,
                "ensemble_accuracy": test_ensemble_result["accuracy"]
            }
        }
        
        with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
            json.dump(comparison_results, f, indent=4)
    
if __name__ == "__main__":
    print(f"Starting evaluation of {n_splits}-fold cross-validation models")
    print(f"Models directory: {model_base_dir}")
    print(f"Output directory: {output_dir}")
    
    main()
    
    print("Evaluation completed!") 