from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import os
import json
import shutil
from scipy.spatial.distance import cosine

@dataclass
class FilterResult:
    """필터링 결과를 담는 데이터 클래스"""
    filtered_images: Dict[str, float]  # {image_path: similarity_score}
    rejected_images: Dict[str, float]  # 유사도 점수도 저장
    statistics: Dict[str, float]

class VectorFilter:
    def __init__(self, 
                 origin_vector_path: str,
                 generated_vector_path: str,
                 generated_dir: str,
                 output_dir: str):
        """
        Args:
            origin_vector_path: 원본 이미지 벡터 경로 (.npy)
            generated_vector_path: 생성된 이미지 벡터 경로 (.npy)
            generated_dir: 생성된 이미지들이 있는 디렉토리 경로
            output_dir: 결과를 저장할 디렉토리 경로
        """
        self.origin_vector_path = origin_vector_path
        self.generated_vector_path = generated_vector_path
        self.generated_dir = generated_dir
        self.output_dir = output_dir
        
        # 벡터 파일 존재 확인
        if not os.path.exists(origin_vector_path):
            raise FileNotFoundError(f"원본 벡터 파일을 찾을 수 없습니다: {origin_vector_path}")
        
        if not os.path.exists(generated_vector_path):
            raise FileNotFoundError(f"생성된 이미지 벡터 파일을 찾을 수 없습니다: {generated_vector_path}")
        
        print(f"Loading vectors...")
        self.origin_vectors = np.load(origin_vector_path, allow_pickle=True).item()
        self.generated_vectors = np.load(generated_vector_path, allow_pickle=True).item()
        
        # 원본 벡터들의 평균 계산
        self.origin_mean = np.mean(list(self.origin_vectors.values()), axis=0)
        print(f"Loaded {len(self.origin_vectors)} original vectors and {len(self.generated_vectors)} generated vectors")
        
        # 필터링 실행
        print("\n=== Starting Filtering ===")
        self.result = self.filter_vectors()
        
        # 결과 저장
        self.save_results()
        
        # 이미지 파일 복사
        self.copy_filtered_images()

    def calculate_similarity(self, vec: np.ndarray) -> float:
        """벡터와 원본 평균 벡터 간의 코사인 유사도 계산"""
        return 1 - cosine(vec, self.origin_mean)

    def filter_vectors(self) -> FilterResult:
        """벡터 필터링 실행"""
        filtered_images = {}
        rejected_images = {}
        
        # 모든 생성된 이미지의 유사도 계산
        all_similarities = []
        for gen_path, gen_vector in self.generated_vectors.items():
            similarity = self.calculate_similarity(gen_vector)
            all_similarities.append(similarity)
        
        # 평균 유사도를 threshold로 사용
        threshold = np.mean(all_similarities)
        print(f"Similarity threshold: {threshold:.4f}")
        
        # threshold를 기준으로 필터링
        for gen_path, gen_vector in self.generated_vectors.items():
            similarity = self.calculate_similarity(gen_vector)
            if similarity >= threshold:
                filtered_images[gen_path] = similarity
            else:
                rejected_images[gen_path] = similarity
        
        stats = self._calculate_statistics(filtered_images, rejected_images, all_similarities)
        
        print("\nFiltering completed!")
        
        return FilterResult(
            filtered_images=filtered_images,
            rejected_images=rejected_images,
            statistics=stats
        )

    def _calculate_statistics(self, 
                            filtered_images: Dict[str, float],
                            rejected_images: Dict[str, float],
                            all_similarities: List[float]) -> Dict[str, float]:
        """필터링 통계 계산"""
        stats = {
            'total_generated': len(self.generated_vectors),
            'filtered_count': len(filtered_images),
            'rejected_count': len(rejected_images),
            'average_similarity': float(np.mean(all_similarities)),
            'threshold': float(np.mean(all_similarities)),
            'min_similarity': float(min(all_similarities)),
            'max_similarity': float(max(all_similarities)),
            'std_similarity': float(np.std(all_similarities))
        }
        
        return stats

    def save_results(self):
        """필터링 결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 텍스트 형식
        with open(os.path.join(self.output_dir, 'filtering_results.txt'), 'w') as f:
            f.write(f"Filtering Results (Threshold: {self.result.statistics['threshold']:.4f})\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Filtered Images\n")
            f.write("-" * 50 + "\n")
            for gen_path, similarity in sorted(self.result.filtered_images.items(), key=lambda x: x[1], reverse=True):
                f.write(f"• {gen_path} (similarity: {similarity:.4f})\n")
            
            f.write("\nRejected Images\n")
            f.write("-" * 50 + "\n")
            for gen_path, similarity in sorted(self.result.rejected_images.items(), key=lambda x: x[1], reverse=True):
                f.write(f"• {gen_path} (similarity: {similarity:.4f})\n")
        
        # 2. JSON 형식
        json_results = {
            "threshold": float(self.result.statistics['threshold']),
            "filtered_images": {
                path: float(sim) for path, sim in self.result.filtered_images.items()
            },
            "rejected_images": {
                path: float(sim) for path, sim in self.result.rejected_images.items()
            },
            "statistics": self.result.statistics
        }
        
        with open(os.path.join(self.output_dir, 'filtering_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        # 3. 필터링 통계 저장
        with open(os.path.join(self.output_dir, 'filtering_stats.txt'), 'w') as f:
            f.write(f"Filtering Statistics (Threshold: {self.result.statistics['threshold']:.4f})\n")
            f.write("=" * 50 + "\n\n")
            for stat_name, stat_value in self.result.statistics.items():
                f.write(f"{stat_name}: {stat_value}\n")
                
        print(f"Results saved to {self.output_dir}")

    def copy_filtered_images(self):
        """필터링된 이미지를 복사"""
        # 필터링된 이미지 저장을 위한 디렉토리 생성
        filtered_dir = os.path.join(self.output_dir, 'filtered_images')
        rejected_dir = os.path.join(self.output_dir, 'rejected_images')
        os.makedirs(filtered_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)
        
        # 필터링된 이미지들 복사 
        print("\nCopying filtered images...")
        for gen_path, similarity in self.result.filtered_images.items():
            # 원본 파일의 전체 경로
            src_path = os.path.join(self.generated_dir, gen_path)
            
            # 새 파일명 생성 (유사도 포함)
            base_name = os.path.splitext(os.path.basename(gen_path))[0]
            new_name = f"{base_name}_sim{similarity:.4f}{os.path.splitext(gen_path)[1]}"
            
            # 대상 파일의 전체 경로
            dst_path = os.path.join(filtered_dir, new_name)
            
            shutil.copy2(src_path, dst_path)
        
        # 거부된 이미지들 복사
        print("Copying rejected images...")
        for gen_path, similarity in self.result.rejected_images.items():
            # 원본 파일의 전체 경로
            src_path = os.path.join(self.generated_dir, gen_path)
            
            # 새 파일명 생성 (유사도 포함)
            base_name = os.path.splitext(os.path.basename(gen_path))[0]
            new_name = f"{base_name}_sim{similarity:.4f}{os.path.splitext(gen_path)[1]}"
            
            # 대상 파일의 전체 경로
            dst_path = os.path.join(rejected_dir, new_name)
            
            shutil.copy2(src_path, dst_path)
            
        print(f"\n=== Processing Completed ===")
        print(f"Filtered images saved to: {filtered_dir}")
        print(f"Rejected images saved to: {rejected_dir}")
        print(f"Total images: {self.result.statistics['total_generated']}")
        print(f"Filtered images: {self.result.statistics['filtered_count']}")
        print(f"Rejected images: {self.result.statistics['rejected_count']}")
        print(f"Average similarity: {self.result.statistics['average_similarity']:.4f}")
        print(f"Threshold: {self.result.statistics['threshold']:.4f}")


def filter_blepharitis():
    """Blepharitis 이미지 필터링"""
    print("\n=== Filtering Blepharitis Images ===")
    # 경로 설정
    vector_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/vectors"
    origin_vector_path = os.path.join(vector_dir, "blepharitis.npy")
    generated_vector_path = os.path.join(vector_dir, "generated_blepharitis.npy")
    
    generated_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/generated/blepharitis"
    output_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/filtered/blepharitis"
    
    # 필터링 실행
    VectorFilter(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        generated_dir=generated_dir,
        output_dir=output_dir
    )


def filter_keratitis():
    """Keratitis 이미지 필터링"""
    print("\n=== Filtering Keratitis Images ===")
    # 경로 설정
    vector_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/vectors"
    origin_vector_path = os.path.join(vector_dir, "keratitis.npy")
    generated_vector_path = os.path.join(vector_dir, "generated_keratitis.npy")
    
    generated_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/generated/keratitis"
    output_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/filtered/keratitis"
    
    # 필터링 실행
    VectorFilter(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        generated_dir=generated_dir,
        output_dir=output_dir
    )



if __name__ == "__main__":
    # 필터링 실행
    try:
        filter_blepharitis()
    except Exception as e:
        print(f"Blepharitis 필터링 중 오류 발생: {e}")
    
    try:
        filter_keratitis()
    except Exception as e:
        print(f"Keratitis 필터링 중 오류 발생: {e}")
    
    print("\n=== 모든 필터링 프로세스 완료 ===")