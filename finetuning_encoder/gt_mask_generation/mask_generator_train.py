from ultralytics import YOLO

# Load a model
model = YOLO("/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/yolo/basic_models/yolo11x-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/yolo_seg_datasets/data.yaml",
    epochs=100,          # 총 에포크 수
    imgsz=640,          # 이미지 크기
    batch=16,           # 배치 크기
    workers=8,          # 데이터 로딩 워커 수
    device=3,           # GPU 선택
    val=True,           # validation 활성화
    save=True,          # 최상의 모델 저장
    save_period=10      # 10 에포크마다 모델 저장
)