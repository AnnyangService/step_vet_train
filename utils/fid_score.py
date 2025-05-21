import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from PIL import Image
import os

class FIDCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Inception v3 모델 로드
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()  # 마지막 fully connected 레이어 제거
        self.model = self.model.to(device)
        self.model.eval()
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_features(self, images):
        features = []
        with torch.no_grad():
            for img in images:
                img = img.unsqueeze(0).to(self.device)
                feature = self.model(img)
                features.append(feature.cpu().numpy())
        return np.concatenate(features, axis=0)

    def calculate_statistics(self, features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(self, real_features, fake_features):
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_fake, sigma_fake = self.calculate_statistics(fake_features)
        
        # FID 계산
        ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
        
        # 복소수 처리
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
        return fid

    def calculate_fid_from_paths(self, real_path, fake_path, batch_size=32):
        # 실제 이미지 로드 및 특징 추출
        real_images = []
        for img_path in os.listdir(real_path):
            if img_path.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(real_path, img_path)).convert('RGB')
                img = self.transform(img)
                real_images.append(img)
        
        # 생성된 이미지 로드 및 특징 추출
        fake_images = []
        for img_path in os.listdir(fake_path):
            if img_path.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(fake_path, img_path)).convert('RGB')
                img = self.transform(img)
                fake_images.append(img)
        
        # 특징 추출
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # FID 계산
        fid_score = self.calculate_fid(real_features, fake_features)
        return fid_score

# 사용 예시
if __name__ == "__main__":
    # FID 계산기 초기화
    fid_calculator = FIDCalculator()
    
    # 실제 이미지와 생성된 이미지 경로
    real_images_path = "/home/minelab/desktop/Jack/step_vet_train/datasets/origin/keratitis"
    fake_images_path = "/home/minelab/desktop/Jack/step_vet_train/datasets/matching_filtered/keratitis/rejected_images"
    
    # FID 스코어 계산
    fid_score = fid_calculator.calculate_fid_from_paths(real_images_path, fake_images_path)
    print(f"FID Score: {fid_score}") 