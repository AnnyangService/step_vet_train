import torch
from transformers import SamModel, SamProcessor
from PIL import Image

class CustomSamEncoder:
    def __init__(self, model_name="facebook/sam-vit-base", checkpoint_path=None, gpu_id=3):
        # GPU 설정
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # SAM 모델과 프로세서 로드
        self.model = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)

        # 체크포인트 로드
        if checkpoint_path:
            self.load_custom_checkpoint(checkpoint_path)

        # 모델을 GPU로 이동
        self.model = self.model.to(self.device)
        
        # 비전 인코더 추출 및 평가 모드 설정
        self.vision_encoder = self.model.vision_encoder
        self.vision_encoder.eval()

    def load_custom_checkpoint(self, checkpoint_path):
        """파인튜닝된 모델의 체크포인트를 로드합니다."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 체크포인트의 vision_encoder 가중치만 로드
        current_state_dict = self.model.state_dict()
        for name, param in checkpoint['vision_encoder_state_dict'].items():
            if 'vision_encoder' in name:
                current_state_dict[name] = param
        
        self.model.load_state_dict(current_state_dict)
        print(f"Loaded custom checkpoint from {checkpoint_path}")

    def extract_features(self, image_path):
        """
        이미지를 입력으로 받아서 인코더를 통과시키고 특징을 추출합니다.
        Args:
            image_path (str): 이미지 파일 경로
        Returns:
            tuple: (features, original_image)
                - features: 인코더 출력값 (torch.Tensor)
                - original_image: 원본 이미지 (PIL.Image)
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
        
        # GPU로 이동
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 인코더 통과
        with torch.no_grad():
            outputs = self.vision_encoder(inputs["pixel_values"])
            features = outputs.last_hidden_state.cpu()
            
        return features, image
