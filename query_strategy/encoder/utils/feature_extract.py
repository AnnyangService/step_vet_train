import torch
from PIL import Image
import sys
sys.path.append('/home/minelab/desktop/')
from Jack.step_vet_train.query_strategy.encoder.utils.sam_encoder_loader import SamEncoderLoader

class FeatureExtractor:
    def __init__(self, model_name="facebook/sam-vit-base", checkpoint_path=None, gpu_id=3):
        # 인코더 로더 초기화
        self.encoder_loader = SamEncoderLoader(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            gpu_id=gpu_id
        )
        
        # 인코더, 프로세서, 디바이스 가져오기
        self.vision_encoder, self.processor, self.device = self.encoder_loader.get_encoder_and_processor()

    def extract_features(self, image_path):
        """이미지에서 특징 추출"""
        try:
            # PIL 이미지로 열기
            image = Image.open(image_path)
            
            # processor 사용 시 명시적 옵션 지정
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={"longest_edge": 1024},
                do_normalize=True
            )
            
            # 텐서만 디바이스로 이동
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.vision_encoder(inputs["pixel_values"])
                image_embeddings = features.last_hidden_state.contiguous()
                return image_embeddings.clone(), image
                
        except Exception as e:
            print(f"Error in feature extraction for {image_path}: {str(e)}")
            return None, None