import torch
from transformers import SamModel, SamProcessor

class SamEncoderLoader:
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

    def get_encoder_and_processor(self):
        """인코더와 프로세서를 반환합니다."""
        return self.vision_encoder, self.processor, self.device
