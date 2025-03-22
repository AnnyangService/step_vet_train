import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from tqdm import tqdm
import argparse

class BlueMaskExtractor:
    """파란색 마스크를 추출하는 클래스"""
    
    def __init__(self, 
                 hue_range: Tuple[int, int] = (100, 140),
                 saturation_range: Tuple[int, int] = (50, 255),
                 value_range: Tuple[int, int] = (50, 255),
                 kernel_size: Tuple[int, int] = (3, 3)):
        """
        Args:
            hue_range: HSV 색상 범위
            saturation_range: HSV 채도 범위
            value_range: HSV 명도 범위
            kernel_size: 노이즈 제거를 위한 커널 크기
        """
        self.lower_blue = np.array([hue_range[0], saturation_range[0], value_range[0]])
        self.upper_blue = np.array([hue_range[1], saturation_range[1], value_range[1]])
        self.kernel = np.ones(kernel_size, np.uint8)

    def extract(self, img: np.ndarray, binary: bool = True) -> np.ndarray:
        """
        이미지에서 파란색 마스크를 추출
        
        Args:
            img: BGR 형식의 입력 이미지
            binary: True면 이진 마스크, False면 원본 마스크 반환
        Returns:
            numpy.ndarray: 마스크 이미지
        """
        # BGR to HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 마스크 생성
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        
        # 노이즈 제거
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, self.kernel)
        
        if binary:
            return (blue_mask > 0).astype(np.uint8)
        return blue_mask

class MaskProcessor:
    """마스크 처리를 위한 유틸리티 클래스"""
    
    @staticmethod
    def resize_mask(mask: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """마스크 크기 조정"""
        return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def save_as_tiff(mask: np.ndarray, output_path: Union[str, Path], size: Optional[Tuple[int, int]] = None) -> None:
        """마스크를 TIFF 형식으로 저장"""
        if size:
            mask = MaskProcessor.resize_mask(mask, size)
        
        pil_mask = Image.fromarray(mask)
        pil_mask.save(str(output_path))

def convert_masks_to_tiff(input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         mask_size: Tuple[int, int] = (256, 256)) -> None:
    """
    디렉토리 내의 모든 jpg 마스크를 tiff로 변환하고 저장
    
    Args:
        input_dir: 입력 디렉토리 경로
        output_dir: 출력 디렉토리 경로
        mask_size: 저장할 마스크 크기
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = BlueMaskExtractor()
    mask_files = list(input_dir.glob("*.jpg"))
    
    for mask_file in tqdm(mask_files, desc="Converting masks"):
        # 출력 파일 경로
        output_path = output_dir / mask_file.with_suffix('.tiff').name
        
        # 이미지 로드 및 마스크 추출
        img = cv2.imread(str(mask_file))
        if img is None:
            print(f"Failed to read image: {mask_file}")
            continue
            
        binary_mask = extractor.extract(img, binary=True)
        
        # TIFF로 저장
        MaskProcessor.save_as_tiff(binary_mask, output_path, size=mask_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='마스크 이미지 TIFF 변환 도구')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='JPG 마스크 이미지가 있는 입력 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='TIFF 마스크를 저장할 출력 디렉토리')
    parser.add_argument('--mask_size', type=int, nargs=2, default=[256, 256],
                        help='출력 마스크 크기 (너비 높이), 기본값: 256 256')
    
    args = parser.parse_args()
    
    print(f"입력 디렉토리: {args.input_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"마스크 크기: {args.mask_size}")
    
    # 마스크 변환 실행
    convert_masks_to_tiff(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mask_size=tuple(args.mask_size)
    )
    
    print("변환 완료!")

