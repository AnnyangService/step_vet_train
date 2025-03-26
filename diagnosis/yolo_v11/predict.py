from ultralytics import YOLO
import cv2
import os

def predict_image(
    model_path,
    image_path,
    conf_threshold=0.25,
    save_results=True
):
    """
    Predict class for a single image using trained YOLO model.
    
    Args:
        model_path (str): Path to the trained YOLO model
        image_path (str): Path to the image file
        conf_threshold (float): Confidence threshold for predictions
        save_results (bool): Whether to save prediction results
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Perform prediction
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=save_results
    )
    
    # Process and print results
    for result in results:
        probs = result.probs
        if probs is not None:
            # 안전하게 top1 값 가져오기 (int인 경우와 tensor인 경우 모두 처리)
            if hasattr(probs.top1, 'item'):
                top1_class = probs.top1.item()
            else:
                top1_class = probs.top1  # 이미 int인 경우
                
            # 안전하게 confidence 값 가져오기
            if hasattr(probs.top1conf, 'item'):
                top1_prob = probs.top1conf.item()
            else:
                top1_prob = probs.top1conf  # 이미 float인 경우
                
            class_name = model.names[top1_class]
            print(f"예측 클래스: {class_name}")
            print(f"신뢰도: {top1_prob:.2f}")
            
            # 모든 클래스에 대한 확률 출력 (상위 3개)
            print("\n상위 예측 결과:")
            probs_dict = {}
            for i, p in enumerate(probs.data.tolist() if hasattr(probs.data, 'tolist') else probs.data):
                probs_dict[model.names[i]] = p
            
            # 상위 3개 클래스 출력
            sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
            for i, (cls_name, prob) in enumerate(sorted_probs[:3]):
                print(f"  {i+1}. {cls_name}: {prob:.4f}")
    
    return results

def predict_directory(
    model_path,
    directory_path,
    conf_threshold=0.25,
    save_results=True
):
    """
    Predict classes for all images in a directory.
    
    Args:
        model_path (str): Path to the trained YOLO model
        directory_path (str): Path to the directory containing images
        conf_threshold (float): Confidence threshold for predictions
        save_results (bool): Whether to save prediction results
    """
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(directory_path) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"경고: {directory_path}에 이미지 파일이 없습니다")
        return []
    
    print(f"총 {len(image_files)}개 이미지 파일을 처리합니다")
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(directory_path, image_file)
        print(f"\n[{i}/{len(image_files)}] 처리 중: {image_file}")
        try:
            result = predict_image(
                model_path=model_path,
                image_path=image_path,
                conf_threshold=conf_threshold,
                save_results=save_results
            )
            results.append(result)
        except Exception as e:
            print(f"이미지 {image_file} 처리 중 오류 발생: {str(e)}")
    
    print(f"\n모든 {len(image_files)}개 이미지 처리 완료")
    return results

if __name__ == "__main__":
    # Configuration
    model_path = "/home/minelab/desktop/Jack/step_vet_train/models/yolo/yolo_v11_cls/train/weights/best.pt"
    image_path = "/home/minelab/desktop/Jack/step_vet_train/datasets/yolo_dataset/test/images/test_image.jpg"
    directory_path = "/home/minelab/desktop/Jack/step_vet_train/datasets/yolo_dataset/test/blepharitis"
    
    # # Predict single image
    # print("단일 이미지 예측 중...")
    # predict_image(
    #     model_path=model_path,
    #     image_path=image_path,
    #     conf_threshold=0.25,
    #     save_results=True
    # )
    
    # Predict all images in directory
    print("\n디렉토리 내 모든 이미지 예측 중...")
    predict_directory(
        model_path=model_path,
        directory_path=directory_path,
        conf_threshold=0.25,
        save_results=True
    ) 