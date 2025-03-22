import os
from PIL import Image 
from datasets import Dataset, Features, Image as ImageFeature

def load_dataset(image_dir: str, mask_dir: str) -> Dataset:
    """
    이미지와 마스크를 Huggingface Dataset 형식으로 로드
    마스크가 없는 이미지는 제외됨
    
    Args:
        image_dir (str): 원본 이미지가 있는 디렉토리 경로
        mask_dir (str): 마스크 이미지가 있는 디렉토리 경로
    
    Returns:
        Dataset: Huggingface Dataset 형식의 데이터셋
    """
    # 이미지 파일 목록 가져오기
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # 데이터 딕셔너리 준비
    data_dict = {
        "image": [],
        "label": []
    }
    
    # 이미지와 마스크 로드
    for img_file in image_files:
        # 대응하는 마스크 파일 경로
        mask_file = os.path.splitext(img_file)[0] + ".tiff"
        mask_path = os.path.join(mask_dir, mask_file)
        
        # 마스크 파일이 존재하는 경우만 처리
        if os.path.exists(mask_path):
            try:
                image = Image.open(os.path.join(image_dir, img_file))
                mask = Image.open(mask_path)
                
                data_dict["image"].append(image)
                data_dict["label"].append(mask)
            except Exception as e:
                print(f"Error loading {img_file}: {str(e)}")
                continue
    
    # Huggingface Dataset 생성
    features = Features({
        "image": ImageFeature(),
        "label": ImageFeature()
    })
    
    return Dataset.from_dict(data_dict, features=features)

# if __name__ == "__main__":
#     # 데이터셋 로드
#     dataset = load_dataset(
#         image_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases",
#         mask_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks/tiff_masks"
#     )
    
#     # 데이터셋 정보 출력
#     print(dataset)
#     print(dataset[0])
#     print(dataset[0]["image"])
#     print(dataset[0]["label"])