from ultralytics import YOLO

def train_model(
    data_path,
    epochs=100,
    batch_size=16,
    imgsz=512,
    project_name="/home/minelab/desktop/Jack/step_vet_train/models/yolo/matching/yolo_v11_cls",
    device=[2, 3]
):
    """
    Train YOLO classification model.
    
    Args:
        weights_path (str): Path to the YOLO weights file
        data_path (str): Path to the dataset directory
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        imgsz (int): Input image size
        project_name (str): Name of the project for saving results
        device (list): List of GPU indices to use
    """
    # Initialize YOLO model
    model = YOLO("yolo11n-cls.pt")
    
    print(f"Starting training with GPUs: {device}")
    print(f"Image size: {imgsz}x{imgsz}")
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=project_name,
        device=device,
        task='classify' 
    )
    
    print("Training completed!")

if __name__ == "__main__":
    # Configuration
    data_path = "/home/minelab/desktop/Jack/step_vet_train/datasets/yolo_dataset"
    
    # Train the model
    train_model(
        data_path=data_path,
        epochs=100,
        batch_size=16,
        imgsz=512,
        device=[2, 3]
    )