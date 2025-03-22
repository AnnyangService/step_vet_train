import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 파인튜닝 전 인코더를 통과시켜 feature map 시각화
class TestSamEncoder:
    def __init__(self, model_name="facebook/sam-vit-huge", gpu_id=3):
        # 디폴트로 3번 GPU 사용
        # GPU 사용 가능 여부 확인 및 device 설정
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # SAM 모델과 프로세서 로드
        self.model = SamModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)  # GPU
        self.processor = SamProcessor.from_pretrained(model_name)

        # 비전 인코더만 추출
        self.vision_encoder = self.model.vision_encoder
        self.vision_encoder.eval()  # 평가 모드로 설정

    def process_image(self, image_path):
        """
        이미지를 입력으로 받아서 인코더 통과시키기
        Args:
            image_path (str): 이미지 파일 경로
        Returns:
            torch.Tensor: 인코더 출력값
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path)
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"longest_edge": 1024},
            do_normalize=True
        )
        
        # GPU
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 인코더 통과       
        with torch.no_grad():
            outputs = self.vision_encoder(inputs["pixel_values"])

            ########## 시각화
            features = outputs.last_hidden_state.cpu()
            # 주의 !!! matplotlib은 CPU 텐서만 처리할 수 있음
            plt.figure(figsize=(15, 5))
            
            # 1. 원본 이미지와 평균 featuremap
            plt.figure(figsize=(15, 5))
        
            plt.subplot(121)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(122)
            feature_mean = features[0].mean(dim=0)
            plt.imshow(feature_mean.numpy(), cmap='viridis')
            plt.title('Average Feature Map')
            plt.colorbar()
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('/home/minelab/desktop/Jack/step_vet_train/visualization/results/encoder/before_finetuning/channels_avg.png')
            plt.close()
            
            # 2. 여러 채널들 시각화 (예: 처음 64개 채널)
            num_channels = 64  # 시각화할 채널 수
            rows = 8
            cols = 8
            plt.figure(figsize=(20, 20))
            
            for i in range(num_channels):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(features[0, i].numpy(), cmap='viridis')
                plt.title(f'Channel {i}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('/home/minelab/desktop/Jack/step_vet_train/visualization/results/encoder/before_finetuning/channels.png')
            
            plt.close()
            
            
        return outputs

if __name__ == "__main__":
    # GPU 설정 및 모델 로드
    encoder = TestSamEncoder(gpu_id=3)
    outputs = encoder.process_image("/home/minelab/desktop/Jack/step_vet_train/visualization/results/encoder/origin.jpg")
    print("Encoder output shape:", outputs.last_hidden_state.shape)
    