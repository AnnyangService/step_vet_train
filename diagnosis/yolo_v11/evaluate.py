from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from datetime import datetime
import platform
import matplotlib as mpl

# 운영체제에 따른 한글 폰트 설정
def set_korean_font():
    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Linux':
        # 리눅스에서 사용할 한글 폰트 설정
        try:
            # NanumGothic 폰트 사용 시도
            plt.rcParams['font.family'] = 'NanumGothic'
        except:
            try:
                # 또는 다른 한글 폰트 시도
                plt.rcParams['font.family'] = 'NanumBarunGothic'
            except:
                print("한글 폰트를 찾을 수 없습니다. 시스템에 한글 폰트를 설치하세요.")
                plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 적용
set_korean_font()

def evaluate_model(
    model_path,
    data_dir,
    imgsz=512,
    batch_size=16,
    device=0,
    save_dir=None
):
    """
    Evaluate YOLO classification model on test dataset.
    
    Args:
        model_path (str): Path to the trained YOLO model
        data_dir (str): Path to the dataset directory (containing test folder)
        imgsz (int): Input image size
        batch_size (int): Batch size for evaluation
        device (int or list): GPU indices to use
        save_dir (str): Directory to save evaluation results
    
    Returns:
        dict: Evaluation metrics
    """
    # Create save directory with timestamp if not provided
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"/home/minelab/desktop/Jack/step_vet_train/results/evaluation_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the trained model
    model = YOLO(model_path)
    print(f"모델 불러오기 완료: {model_path}")
    
    # Run validation on test data
    print(f"테스트 데이터셋 평가 중: {data_dir}")
    print(f"이미지 크기: {imgsz}x{imgsz}, 배치 크기: {batch_size}, 장치: {device}")
    
    results = model.val(
        data=data_dir,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        split='test',
        project=save_dir,
        name='results',
        plots=True  # Generate confusion matrix and other plots
    )
    
    # Print comprehensive metrics
    print("\n" + "="*50)
    print("평가 결과")
    print("="*50)
    
    # 메트릭 저장을 위한 딕셔너리
    metrics_dict = {}
    
    # 정확도 가져오기
    if hasattr(results, 'top1') and hasattr(results, 'top5'):
        top1 = float(results.top1)
        top5 = float(results.top5)
        print(f"Top-1 정확도: {top1:.4f}")
        print(f"Top-5 정확도: {top5:.4f}")
        metrics_dict['Top-1 정확도'] = top1
        metrics_dict['Top-5 정확도'] = top5
    
    # YOLO의 컨퓨전 매트릭스 확인
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        cm = results.confusion_matrix
        if isinstance(cm, np.ndarray):
            classes = model.names
            
            # 컨퓨전 매트릭스에서 메트릭 계산
            # 전체 정확도 계산
            total_correct = np.sum(np.diag(cm))
            total_samples = np.sum(cm)
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            
            # 각 클래스별 메트릭 계산
            class_metrics = []
            precision_values = []
            recall_values = []
            f1_values = []
            
            print("\n" + "-"*50)
            print("클래스별 평가 지표")
            print("-"*50)
            print(f"{'클래스명':<20} {'정밀도':<10} {'재현율':<10} {'F1 점수':<10}")
            print("-"*50)
            
            for i, class_name in enumerate(classes):
                # 해당 클래스의 TP, FP, FN 계산
                tp = cm[i, i]  # 대각선 값 (True Positive)
                fp = np.sum(cm[:, i]) - tp  # 해당 열의 합에서 TP 빼기 (False Positive)
                fn = np.sum(cm[i, :]) - tp  # 해당 행의 합에서 TP 빼기 (False Negative)
                
                # 정밀도(Precision): TP / (TP + FP)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                precision_values.append(precision)
                
                # 재현율(Recall): TP / (TP + FN)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                recall_values.append(recall)
                
                # F1 점수: 2 * (precision * recall) / (precision + recall)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_values.append(f1)
                
                # 클래스별 지표 저장
                class_metrics.append({
                    '클래스': class_name,
                    '정밀도': precision,
                    '재현율': recall,
                    'F1 점수': f1
                })
                
                # 표 형식으로 출력
                print(f"{class_name:<20} {precision:.4f}     {recall:.4f}     {f1:.4f}")
            
            # 매크로 평균 계산
            macro_precision = np.mean(precision_values)
            macro_recall = np.mean(recall_values)
            macro_f1 = np.mean(f1_values)
            
            # 전체 지표 저장
            metrics_dict['정확도'] = accuracy
            metrics_dict['매크로 정밀도'] = macro_precision
            metrics_dict['매크로 재현율'] = macro_recall
            metrics_dict['매크로 F1 점수'] = macro_f1
            
            # 전체 성능 출력
            print("\n" + "-"*50)
            print("전체 평가 지표")
            print("-"*50)
            print(f"전체 정확도(Accuracy): {accuracy:.4f}")
            print(f"매크로 정밀도(Precision): {macro_precision:.4f}")
            print(f"매크로 재현율(Recall): {macro_recall:.4f}")
            print(f"매크로 F1 점수(F1-Score): {macro_f1:.4f}")
            print("-"*50)
            
            # 메트릭을 CSV 파일로 저장
            class_df = pd.DataFrame(class_metrics)
            class_csv_path = os.path.join(save_dir, 'class_metrics.csv')
            class_df.to_csv(class_csv_path, index=False, encoding='utf-8-sig')
            
            overall_df = pd.DataFrame([metrics_dict])
            overall_csv_path = os.path.join(save_dir, 'overall_metrics.csv')
            overall_df.to_csv(overall_csv_path, index=False, encoding='utf-8-sig')
            
            print(f"\n클래스별 메트릭 저장 완료: {class_csv_path}")
            print(f"전체 메트릭 저장 완료: {overall_csv_path}")
            
            # 컨퓨전 매트릭스 시각화
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.xlabel('예측 클래스')
            plt.ylabel('실제 클래스')
            plt.title('컨퓨전 매트릭스')
            plt.tight_layout()
            confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
            plt.savefig(confusion_matrix_path, dpi=300)
            print(f"컨퓨전 매트릭스 저장 완료: {confusion_matrix_path}")
            
            # 정규화된 컨퓨전 매트릭스
            plt.figure(figsize=(12, 10))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.xlabel('예측 클래스')
            plt.ylabel('실제 클래스')
            plt.title('정규화된 컨퓨전 매트릭스')
            plt.tight_layout()
            normalized_confusion_matrix_path = os.path.join(save_dir, 'normalized_confusion_matrix.png')
            plt.savefig(normalized_confusion_matrix_path, dpi=300)
            print(f"정규화된 컨퓨전 매트릭스 저장 완료: {normalized_confusion_matrix_path}")
            
            # 성능 시각화 - 클래스별 정밀도, 재현율, F1 점수
            plt.figure(figsize=(14, 8))
            x = np.arange(len(classes))
            width = 0.25
            
            plt.bar(x - width, precision_values, width, label='정밀도(Precision)')
            plt.bar(x, recall_values, width, label='재현율(Recall)')
            plt.bar(x + width, f1_values, width, label='F1 점수(F1-Score)')
            
            plt.xlabel('클래스')
            plt.ylabel('점수')
            plt.title('클래스별 성능 지표')
            plt.xticks(x, classes, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            performance_metrics_path = os.path.join(save_dir, 'class_performance.png')
            plt.savefig(performance_metrics_path, dpi=300)
            print(f"클래스별 성능 지표 시각화 저장 완료: {performance_metrics_path}")
            
            print(f"\n모든 평가 결과가 저장된 경로: {save_dir}")
    else:
        print("컨퓨전 매트릭스를 찾을 수 없습니다.")
    
    return results

if __name__ == "__main__":
    # Configuration
    model_path = "/home/minelab/desktop/Jack/step_vet_train/models/yolo/matching/yolo_v11_cls_anomaly_detection/train/weights/best.pt"
    data_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/yolo_dataset"
    
    # Evaluate the model
    results = evaluate_model(
        model_path=model_path,
        data_dir=data_dir,
        imgsz=512,
        batch_size=16,
        device=0
    ) 