from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import os
import json
import shutil
from scipy.spatial.distance import cosine

@dataclass
class MatchingResult:
    """이미지 매칭 결과를 담는 데이터 클래스"""
    matched_pairs: Dict[str, List[Tuple[str, float]]]  # {origin_image: [(matched_image, similarity), ...]}
    unmatched_images: List[str]  # 매칭되지 않은 생성 이미지들
    statistics: Dict[str, float]

class ImageMatcher:
    def __init__(self, 
                 origin_vector_path: str,
                 generated_vector_path: str,
                 generated_dir: str,
                 output_dir: str,
                 similarity_threshold: float = 0.7,
                 matches_per_image: int = 2):
        """
        Args:
            origin_vector_path: 원본 이미지 벡터 경로 (.npy)
            generated_vector_path: 생성된 이미지 벡터 경로 (.npy)
            generated_dir: 생성된 이미지들이 있는 디렉토리 경로
            output_dir: 결과를 저장할 디렉토리 경로
            similarity_threshold: 매칭으로 간주할 최소 유사도 값
            matches_per_image: 각 원본 이미지당 매칭할 생성 이미지 수
        """
        self.origin_vector_path = origin_vector_path
        self.generated_vector_path = generated_vector_path
        self.generated_dir = generated_dir
        self.output_dir = output_dir
        self.similarity_threshold = similarity_threshold
        self.matches_per_image = matches_per_image
        
        # output dir 확인 및 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 벡터 파일 존재 확인
        if not os.path.exists(origin_vector_path):
            raise FileNotFoundError(f"원본 벡터 파일을 찾을 수 없습니다: {origin_vector_path}")
        
        if not os.path.exists(generated_vector_path):
            raise FileNotFoundError(f"생성된 이미지 벡터 파일을 찾을 수 없습니다: {generated_vector_path}")
        
        print(f"Loading vectors...")
        self.origin_vectors = np.load(origin_vector_path, allow_pickle=True).item()
        self.generated_vectors = np.load(generated_vector_path, allow_pickle=True).item()
        print(f"Loaded {len(self.origin_vectors)} original vectors and {len(self.generated_vectors)} generated vectors")
        
        # 매칭 실행
        print("\n=== Starting Image Matching ===")
        self.result = self.match_images()
        
        # 결과 저장
        self.save_results()
        
        # 매칭된 이미지 파일 복사
        self.copy_matched_images()

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        return 1 - cosine(vec1, vec2)

    def match_images(self) -> MatchingResult:
        """이미지 매칭 실행"""
        matched_pairs = {}
        used_generated_images = set()
        
        # 각 원본 이미지에 대해 가장 유사한 생성 이미지 찾기
        for origin_path, origin_vector in self.origin_vectors.items():
            # 각 생성 이미지와의 유사도 계산
            similarities = []
            for gen_path, gen_vector in self.generated_vectors.items():
                similarity = self.calculate_similarity(origin_vector, gen_vector)
                similarities.append((gen_path, similarity))
            
            # 유사도 내림차순 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 각 원본 이미지에 대해 상위 N개 매칭
            matches = []
            for gen_path, similarity in similarities:
                if similarity >= self.similarity_threshold and len(matches) < self.matches_per_image:
                    matches.append((gen_path, similarity))
                    used_generated_images.add(gen_path)
                    print(f"Matched: {origin_path} -> {gen_path} (similarity: {similarity:.4f})")
            
            matched_pairs[origin_path] = matches
        
        # 매칭되지 않은 생성 이미지들
        unmatched_images = [
            path for path in self.generated_vectors.keys() 
            if path not in used_generated_images
        ]
        
        for img in unmatched_images:
            print(f"Unmatched: {img} (below threshold: {self.similarity_threshold})")
        
        # 통계 계산
        total_matches = sum(len(matches) for matches in matched_pairs.values())
        avg_matches_per_origin = total_matches / len(self.origin_vectors) if self.origin_vectors else 0
        
        # 모든 매칭 쌍의 유사도 값
        all_similarities = [
            sim for matches in matched_pairs.values() 
            for _, sim in matches
        ]
        
        stats = {
            'total_origin_images': len(self.origin_vectors),
            'total_generated_images': len(self.generated_vectors),
            'total_matches': total_matches,
            'unmatched_images': len(unmatched_images),
            'avg_matches_per_origin': float(avg_matches_per_origin),
            'threshold': float(self.similarity_threshold),
            'avg_similarity': float(np.mean(all_similarities)) if all_similarities else 0.0,
            'min_similarity': float(min(all_similarities)) if all_similarities else 0.0,
            'max_similarity': float(max(all_similarities)) if all_similarities else 0.0
        }
        
        print("\nMatching completed!")
        
        return MatchingResult(
            matched_pairs=matched_pairs,
            unmatched_images=unmatched_images,
            statistics=stats
        )

    def save_results(self):
        """매칭 결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 텍스트 형식
        with open(os.path.join(self.output_dir, 'matching_results.txt'), 'w') as f:
            f.write(f"Image Matching Results (Threshold: {self.similarity_threshold:.4f})\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Matched Pairs\n")
            f.write("-" * 60 + "\n")
            for origin_path, matches in self.result.matched_pairs.items():
                f.write(f"Original: {origin_path}\n")
                if matches:
                    for gen_path, similarity in matches:
                        f.write(f"  → Matched: {gen_path} (similarity: {similarity:.4f})\n")
                else:
                    f.write("  → No matches found above threshold\n")
                f.write("\n")
            
            f.write("\nUnmatched Images\n")
            f.write("-" * 60 + "\n")
            for gen_path in self.result.unmatched_images:
                f.write(f"• {gen_path}\n")
        
        # 2. JSON 형식
        json_results = {
            "threshold": float(self.similarity_threshold),
            "matched_pairs": {
                origin_path: [(gen_path, float(sim)) for gen_path, sim in matches]
                for origin_path, matches in self.result.matched_pairs.items()
            },
            "unmatched_images": self.result.unmatched_images,
            "statistics": self.result.statistics
        }
        
        with open(os.path.join(self.output_dir, 'matching_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        # 3. 매칭 통계 저장
        with open(os.path.join(self.output_dir, 'matching_stats.txt'), 'w') as f:
            f.write(f"Matching Statistics (Threshold: {self.similarity_threshold:.4f})\n")
            f.write("=" * 60 + "\n\n")
            for stat_name, stat_value in self.result.statistics.items():
                f.write(f"{stat_name}: {stat_value}\n")
                
        print(f"Results saved to {self.output_dir}")

    def copy_matched_images(self):
        """매칭된 이미지를 복사"""
        # 필터링된 이미지 저장을 위한 디렉토리 생성
        filtered_dir = os.path.join(self.output_dir, 'filtered_images')
        rejected_dir = os.path.join(self.output_dir, 'rejected_images')
        os.makedirs(filtered_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)
        
        # 사용된 생성 이미지를 추적하기 위한 집합
        used_generated_images = set()
        
        # 매칭된 이미지 복사
        print("\nCopying matched images...")
        for origin_path, matches in self.result.matched_pairs.items():
            for gen_path, similarity in matches:
                # 원본 파일의 전체 경로
                src_path = os.path.join(self.generated_dir, gen_path)
                
                # 원래 파일명 유지하고 유사도 추가
                filename = os.path.basename(gen_path)
                name, ext = os.path.splitext(filename)
                new_name = f"{name}_sim{similarity:.4f}{ext}"
                
                # 대상 파일의 전체 경로
                dst_path = os.path.join(filtered_dir, new_name)
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    used_generated_images.add(gen_path)
                    print(f"Copied matched image: {gen_path} → {dst_path} (similarity: {similarity:.4f})")
                else:
                    print(f"Warning: Source file not found: {src_path}")
        
        # 거부된/매칭되지 않은 이미지들 복사
        print("\nCopying rejected images...")
        for gen_path in self.result.unmatched_images:
            # 원본 파일의 전체 경로
            src_path = os.path.join(self.generated_dir, gen_path)
            
            # 원래 파일명 유지
            filename = os.path.basename(gen_path)
            
            # 대상 파일의 전체 경로
            dst_path = os.path.join(rejected_dir, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied rejected image: {gen_path}")
            else:
                print(f"Warning: Source file not found: {src_path}")
            
        print(f"\n=== Processing Completed ===")
        print(f"Filtered images saved to: {filtered_dir}")
        print(f"Rejected images saved to: {rejected_dir}")
        print(f"Total original images: {self.result.statistics['total_origin_images']}")
        print(f"Total generated images: {self.result.statistics['total_generated_images']}")
        print(f"Total matches: {self.result.statistics['total_matches']}")
        print(f"Unmatched images: {self.result.statistics['unmatched_images']}")
        print(f"Average matches per original image: {self.result.statistics['avg_matches_per_origin']:.2f}")
        print(f"Average similarity: {self.result.statistics['avg_similarity']:.4f}")
        print(f"Threshold: {self.result.statistics['threshold']:.4f}")


def match_blepharitis(similarity_threshold=0.7):
    """Blepharitis 이미지 매칭"""
    print("\n=== Matching Blepharitis Images ===")
    # 경로 설정
    vector_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/vectors"
    origin_vector_path = os.path.join(vector_dir, "blepharitis.npy")
    generated_vector_path = os.path.join(vector_dir, "generated_blepharitis.npy")
    
    generated_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/generated/blepharitis"
    output_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/matching_filtered/blepharitis"
    
    # 매칭 실행
    ImageMatcher(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        generated_dir=generated_dir,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold
    )


def match_keratitis(similarity_threshold=0.7):
    """Keratitis 이미지 매칭"""
    print("\n=== Matching Keratitis Images ===")
    # 경로 설정
    vector_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/vectors"
    origin_vector_path = os.path.join(vector_dir, "keratitis.npy")
    generated_vector_path = os.path.join(vector_dir, "generated_keratitis.npy")
    
    generated_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/generated/keratitis"
    output_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/matching_filtered/keratitis"
    
    # 매칭 실행
    ImageMatcher(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        generated_dir=generated_dir,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold
    )


if __name__ == "__main__":
    # 매칭 실행 (유사도 임계값 0.7로 설정)
    threshold = 0.7
    
    try:
        match_blepharitis(similarity_threshold=threshold)
    except Exception as e:
        print(f"Blepharitis 매칭 중 오류 발생: {e}")
    
    try:
        match_keratitis(similarity_threshold=threshold)
    except Exception as e:
        print(f"Keratitis 매칭 중 오류 발생: {e}")
    
    print("\n=== 모든 매칭 프로세스 완료 ===")
