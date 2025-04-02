import cv2
import numpy as np
import shutil
from pathlib import Path
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional
from Jack.step_vet_train.finetuning_encoder.utils.mask_processor import BlueMaskExtractor

@dataclass
class MaskQualityMetrics:
    """마스크 품질 평가 메트릭을 저장하는 데이터 클래스"""
    area_ratio: float
    largest_component_ratio: float
    total_area: int
    
    def is_valid(self, 
                 max_area_threshold: float,
                 min_area_threshold: float,
                 min_component_ratio: float) -> Tuple[bool, str]:
        """
        마스크가 유효한지 검사하고 결과와 사유를 반환
        """
        if self.area_ratio == 0:
            return False, "No mask detected"
        if self.area_ratio > max_area_threshold:
            return False, f"Mask too large: {self.area_ratio:.3f}"
        if self.area_ratio < min_area_threshold:
            return False, f"Mask too small: {self.area_ratio:.3f}"
        if self.largest_component_ratio < min_component_ratio:
            return False, f"Multiple separated regions: {self.largest_component_ratio:.3f}"
        return True, "Good quality mask"

class MaskQualityAnalyzer:
    """마스크 품질을 분석하는 클래스"""
    
    @staticmethod
    def evaluate_mask(mask: np.ndarray) -> MaskQualityMetrics:
        """
        마스크의 품질을 평가하여 메트릭을 반환
        """
        if np.sum(mask) == 0:
            return MaskQualityMetrics(0, 0, 0)
            
        mask_area_ratio = np.sum(mask > 0) / mask.size
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        
        largest_component_ratio = (np.max(component_sizes) / np.sum(mask > 0) 
                                 if len(component_sizes) > 0 else 0)
        
        return MaskQualityMetrics(
            area_ratio=mask_area_ratio,
            largest_component_ratio=largest_component_ratio,
            total_area=np.sum(mask > 0)
        )

class MaskClassifier:
    """마스크 이미지를 품질에 따라 분류하는 클래스"""
    
    def __init__(self,
                 output_dir: str,
                 max_area_threshold: float = 0.8,
                 min_area_threshold: float = 0.01,
                 min_component_ratio: float = 0.85):
        """
        Args:
            output_dir: 결과 저장 디렉토리
            max_area_threshold: 최대 마스크 면적 비율
            min_area_threshold: 최소 마스크 면적 비율
            min_component_ratio: 최소 연결 컴포넌트 비율
        """
        self.output_base = Path(output_dir)
        self.max_area_threshold = max_area_threshold
        self.min_area_threshold = min_area_threshold
        self.min_component_ratio = min_component_ratio
        
        self.mask_extractor = BlueMaskExtractor()
        self.quality_analyzer = MaskQualityAnalyzer()
        
        self._setup_directories()
        
    def _setup_directories(self):
        """결과 저장을 위한 디렉토리 구조 생성"""
        self.good_dir = self.output_base / "good_masks"
        self.poor_dir = self.output_base / "poor_masks"
        
        # 불량 마스크 세부 분류 디렉토리
        self.poor_large_dir = self.poor_dir / "too_large"
        self.poor_small_dir = self.poor_dir / "too_small"
        self.poor_split_dir = self.poor_dir / "split_regions"
        self.poor_empty_dir = self.poor_dir / "no_mask"
        
        # 모든 디렉토리 생성
        for dir_path in [self.good_dir, self.poor_large_dir, 
                        self.poor_small_dir, self.poor_split_dir, 
                        self.poor_empty_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def classify_masks(self, pred_dir: str):
        """
        디렉토리 내의 모든 마스크 이미지를 분류
        
        Args:
            pred_dir: 분류할 마스크 이미지가 있는 디렉토리
        """
        pred_dir = Path(pred_dir)
        log_file = self.output_base / "classification_log.csv"
        
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "filename", "result", 
                "area_ratio", "largest_component_ratio", 
                "total_area", "destination"
            ])
            
            for pred_path in pred_dir.glob("*"):
                if pred_path.is_file() and pred_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self._process_single_image(pred_path, writer)
    
    def _process_single_image(self, image_path: Path, csv_writer):
        """단일 이미지 처리 및 결과 기록"""
        try:
            # 이미지 읽기
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Failed to read image: {image_path.name}")
                return
            
            # 마스크 추출 및 품질 평가
            mask = self.mask_extractor.extract(img)
            metrics = self.quality_analyzer.evaluate_mask(mask)
            
            # 품질 판정
            is_good, reason = metrics.is_valid(
                self.max_area_threshold,
                self.min_area_threshold,
                self.min_component_ratio
            )
            
            # 저장 위치 결정
            if metrics.area_ratio == 0:
                dest_dir = self.poor_empty_dir
            elif metrics.area_ratio > self.max_area_threshold:
                dest_dir = self.poor_large_dir
            elif metrics.area_ratio < self.min_area_threshold:
                dest_dir = self.poor_small_dir
            elif metrics.largest_component_ratio < self.min_component_ratio:
                dest_dir = self.poor_split_dir
            else:
                dest_dir = self.good_dir
            
            # 결과 저장
            shutil.copy2(image_path, dest_dir / image_path.name)
            
            # 로그 기록
            csv_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_path.name,
                reason,
                f"{metrics.area_ratio:.3f}",
                f"{metrics.largest_component_ratio:.3f}",
                metrics.total_area,
                dest_dir.name
            ])
            
            print(f"Processed {image_path.name}")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")

if __name__ == "__main__":
    classifier = MaskClassifier(
        output_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks",
        max_area_threshold=0.8,
        min_area_threshold=0.01,
        min_component_ratio=0.85
    )
    
    classifier.classify_masks(
        "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks_origin"
    )