from ultralytics import YOLO

# Load a model
model = YOLO("/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/yolo/custom_models/best.pt")  # load an official model

# Predict with the model
# 예측된 결과는 gt 마스크로 사용
results = model.predict(
    source="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases",
    save=True,  # 결과 이미지 저장
    project="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks",  # 저장될 디렉토리
    boxes=False,
    conf=0.3,
    retina_masks=True,  # 더 부드러운 마스크 윤곽선
    show_labels=False,  # 레이블 텍스트 제거
)