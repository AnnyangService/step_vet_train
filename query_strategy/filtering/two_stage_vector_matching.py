from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import numpy as np
import os
import json
import shutil
from scipy.spatial.distance import cosine
import time
from tqdm import tqdm

# 로그 출력을 제어하기 위한 변수
VERBOSE = False  # 상세 로그 출력 여부

@dataclass
class MatchingResult:
    """이미지 매칭 결과를 담는 데이터 클래스"""
    matched_pairs: Dict[str, List[Tuple[str, float, Dict[str, float]]]]  # {origin_image: [(matched_image, similarity, other_class_scores), ...]}
    unmatched_images: List[str]  # 매칭되지 않은 생성 이미지들
    statistics: Dict[str, float]

class TwoStageImageMatcher:
    def __init__(self, 
                 origin_vector_path: str,
                 generated_vector_path: str,
                 other_class_vector_paths: Dict[str, str],  # {class_name: vector_path}
                 generated_dir: str,
                 output_dir: str,
                 similarity_threshold: float = 0.7,
                 dissimilarity_threshold: float = 0.5,
                 matches_per_image: int = 1,
                 first_stage_keep_ratio: float = 0.3):  # 1단계에서 유지할 비율
        """
        Args:
            origin_vector_path: 원본 이미지 벡터 경로 (.npy)
            generated_vector_path: 생성된 이미지 벡터 경로 (.npy)
            other_class_vector_paths: 다른 클래스 벡터 경로 딕셔너리 {class_name: vector_path}
            generated_dir: 생성된 이미지들이 있는 디렉토리 경로
            output_dir: 결과를 저장할 디렉토리 경로
            similarity_threshold: 매칭으로 간주할 최소 유사도 값
            dissimilarity_threshold: 다른 클래스와 최대 유사도 제한
            matches_per_image: 각 원본 이미지당 매칭할 생성 이미지 수
            first_stage_keep_ratio: 1단계에서 유지할 상위 이미지 비율 (0-1)
        """
        self.origin_vector_path = origin_vector_path
        self.generated_vector_path = generated_vector_path
        self.other_class_vector_paths = other_class_vector_paths
        self.generated_dir = generated_dir
        self.output_dir = output_dir
        self.similarity_threshold = similarity_threshold
        self.dissimilarity_threshold = dissimilarity_threshold
        self.matches_per_image = matches_per_image
        self.first_stage_keep_ratio = first_stage_keep_ratio
        
        # output dir 확인 및 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 벡터 파일 존재 확인
        self._check_vector_files()
        
        # 벡터 로드
        self._load_vectors()
        
        # 매칭 실행
        print("\n=== Starting Two-Stage Multi-Class Image Matching ===")
        start_time = time.time()
        self.result = self.match_images_two_stage()
        elapsed_time = time.time() - start_time
        print(f"Matching completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        # 결과 저장
        self.save_results()
        
        # 매칭된 이미지 파일 복사
        self.copy_matched_images()

    def _check_vector_files(self):
        """벡터 파일 존재 확인"""
        if not os.path.exists(self.origin_vector_path):
            raise FileNotFoundError(f"원본 벡터 파일을 찾을 수 없습니다: {self.origin_vector_path}")
        
        if not os.path.exists(self.generated_vector_path):
            raise FileNotFoundError(f"생성된 이미지 벡터 파일을 찾을 수 없습니다: {self.generated_vector_path}")
        
        # 다른 클래스 벡터 파일 확인
        for class_name, vector_path in self.other_class_vector_paths.items():
            if not os.path.exists(vector_path):
                raise FileNotFoundError(f"{class_name} 클래스의 벡터 파일을 찾을 수 없습니다: {vector_path}")

    def _load_vectors(self):
        """벡터 데이터 로드"""
        print(f"Loading vectors...")
        self.origin_vectors = np.load(self.origin_vector_path, allow_pickle=True).item()
        self.generated_vectors = np.load(self.generated_vector_path, allow_pickle=True).item()
        
        # 다른 클래스 벡터 로드
        self.other_class_vectors = {}
        for class_name, vector_path in self.other_class_vector_paths.items():
            self.other_class_vectors[class_name] = np.load(vector_path, allow_pickle=True).item()
            print(f"Loaded {len(self.other_class_vectors[class_name])} vectors for class: {class_name}")
        
        print(f"Loaded {len(self.origin_vectors)} original vectors and {len(self.generated_vectors)} generated vectors")

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        return 1 - cosine(vec1, vec2)

    def calculate_class_similarities(self, vector: np.ndarray) -> Dict[str, float]:
        """벡터와 모든 다른 클래스 간의 최대 유사도 계산"""
        class_similarities = {}
        for class_name, class_vectors in self.other_class_vectors.items():
            max_similarity = self._calculate_max_similarity_to_class(vector, class_vectors)
            class_similarities[class_name] = max_similarity
        return class_similarities

    def _calculate_max_similarity_to_class(self, vector: np.ndarray, class_vectors: Dict[str, np.ndarray]) -> float:
        """벡터와 클래스 간의 최대 유사도 계산"""
        similarities = [self.calculate_similarity(vector, class_vec) for class_vec in class_vectors.values()]
        return max(similarities) if similarities else 0.0

    def calculate_multi_class_score(self, origin_sim: float, other_class_similarities: Dict[str, float]) -> float:
        """멀티클래스 스코어 계산 - 원본 유사도와 다른 클래스와의 비유사성을 고려"""
        # 다른 클래스와의 최대 유사도
        max_other_sim = max(other_class_similarities.values()) if other_class_similarities else 0
        
        # 원본 유사도 - 다른 클래스 최대 유사도
        # 이 값이 클수록 원본과는 유사하고 다른 클래스와는 차이가 큼을 의미
        return origin_sim - max_other_sim

    def match_images_two_stage(self) -> MatchingResult:
        """두 단계 매칭 방식으로 이미지 매칭 실행"""
        matched_pairs = {}
        used_generated_images = set()
        
        # 진행 상황 추적을 위한 변수
        total_origins = len(self.origin_vectors)
        print(f"Processing {total_origins} original images with two-stage matching...")
        
        # 1단계: 각 원본 이미지에 대해 기본 유사도 기준으로 후보 선별
        print("=== Stage 1: Initial Filtering ===")
        stage1_results = self._stage1_filtering()
        
        # 2단계: 후보 이미지들에 대해 다른 클래스와의 유사도 계산 및 최종 매칭
        print("\n=== Stage 2: Multi-Class Scoring ===")
        matched_pairs = self._stage2_scoring(stage1_results, used_generated_images)
        
        # 매칭되지 않은 생성 이미지들
        unmatched_images = [
            path for path in self.generated_vectors.keys() 
            if path not in used_generated_images
        ]
        
        print(f"\nUnmatched images: {len(unmatched_images)}")
        
        # 통계 계산
        stats = self._calculate_statistics(matched_pairs, unmatched_images)
        
        print("\nTwo-stage matching completed!")
        
        return MatchingResult(
            matched_pairs=matched_pairs,
            unmatched_images=unmatched_images,
            statistics=stats
        )
    
    def _stage1_filtering(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        1단계: 각 원본 이미지에 대해 기본 유사도 기준으로 상위 N% 후보 선별
        """
        stage1_results = {}
        
        print(f"Stage 1: Keeping top {self.first_stage_keep_ratio*100:.1f}% of similar images for each original")
        
        # 원본 이미지를 tqdm으로 진행 상황 표시
        for origin_path, origin_vector in tqdm(self.origin_vectors.items(), total=len(self.origin_vectors), desc="Stage 1"):
            # 각 생성 이미지와의 유사도 계산
            similarities = []
            
            # 일괄 계산 방식으로 진행 - 모든 생성 이미지에 대한 유사도 계산
            for gen_path, gen_vector in self.generated_vectors.items():
                similarity = self.calculate_similarity(origin_vector, gen_vector)
                
                # 기본 유사도 기준을 충족하면 후보로 저장
                if similarity >= self.similarity_threshold:
                    similarities.append((gen_path, similarity))
            
            # 유사도로 내림차순 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 N% 후보만 유지
            keep_count = max(1, int(len(similarities) * self.first_stage_keep_ratio))
            stage1_results[origin_path] = similarities[:keep_count]
            
            # 100개 단위로 진행 상황 출력 (출력 줄이기)
            if len(stage1_results) % 100 == 0:
                print(f"  Processed {len(stage1_results)}/{len(self.origin_vectors)} original images")
        
        total_candidates = sum(len(candidates) for candidates in stage1_results.values())
        avg_candidates = total_candidates / len(stage1_results) if stage1_results else 0
        
        print(f"Stage 1 completed: {total_candidates} total candidates selected")
        print(f"Average {avg_candidates:.1f} candidates per original image")
        
        return stage1_results
    
    def _stage2_scoring(self, stage1_results: Dict[str, List[Tuple[str, float]]], used_generated_images: Set[str]) -> Dict[str, List[Tuple[str, float, Dict[str, float]]]]:
        """
        2단계: 1단계에서 선별된 후보들에 대해 다른 클래스와의 유사도 계산 및 최종 매칭
        """
        matched_pairs = {}
        total_stage1_candidates = sum(len(candidates) for candidates in stage1_results.values())
        
        # 매칭 결과 요약을 위한 카운터
        matched_count = 0
        
        # 2단계 진행 상황 표시
        with tqdm(total=total_stage1_candidates, desc="Stage 2") as pbar:
            for origin_path, candidates in stage1_results.items():
                origin_vector = self.origin_vectors[origin_path]
                
                # 후보들에 대해 멀티클래스 스코어 계산
                final_candidates = []
                
                for gen_path, origin_similarity in candidates:
                    # 이미 사용된 이미지는 건너뛰기
                    if gen_path in used_generated_images:
                        pbar.update(1)
                        continue
                    
                    # 생성 이미지 벡터
                    gen_vector = self.generated_vectors[gen_path]
                    
                    # 다른 클래스와의 유사도 계산
                    other_class_similarities = self.calculate_class_similarities(gen_vector)
                    
                    # 다른 클래스와 너무 유사한 이미지는 제외
                    if any(sim > self.dissimilarity_threshold for sim in other_class_similarities.values()):
                        pbar.update(1)
                        continue
                    
                    # 멀티클래스 스코어 계산
                    multi_class_score = self.calculate_multi_class_score(origin_similarity, other_class_similarities)
                    
                    final_candidates.append((gen_path, origin_similarity, other_class_similarities, multi_class_score))
                    pbar.update(1)
                
                # 멀티클래스 스코어로 내림차순 정렬
                final_candidates.sort(key=lambda x: x[3], reverse=True)
                
                # 각 원본 이미지에 대해 상위 N개 매칭
                matches = []
                for gen_path, origin_similarity, other_class_similarities, score in final_candidates[:self.matches_per_image]:
                    matches.append((gen_path, origin_similarity, other_class_similarities))
                    used_generated_images.add(gen_path)
                    matched_count += 1
                    
                    # 매칭 정보 출력 (상세 로그를 원할 경우에만)
                    if VERBOSE:
                        print(f"Matched: {origin_path} -> {gen_path}")
                        print(f"  Origin similarity: {origin_similarity:.4f}")
                        print(f"  Other class similarities: {', '.join([f'{k}: {v:.4f}' for k, v in other_class_similarities.items()])}")
                        print(f"  Multi-class score: {score:.4f}")
                
                matched_pairs[origin_path] = matches
                
                # 100개 단위로 진행 상황 요약 출력
                if len(matched_pairs) % 100 == 0:
                    print(f"  Matched {len(matched_pairs)}/{len(stage1_results)} original images, total matches: {matched_count}")
        
        print(f"Stage 2 completed: {matched_count} total matches found")
        return matched_pairs
    
    def _calculate_statistics(self, matched_pairs: Dict[str, List[Tuple[str, float, Dict[str, float]]]], unmatched_images: List[str]) -> Dict[str, float]:
        """매칭 통계 계산"""
        # 통계 계산
        total_matches = sum(len(matches) for matches in matched_pairs.values())
        avg_matches_per_origin = total_matches / len(self.origin_vectors) if self.origin_vectors else 0
        
        # 모든 매칭 쌍의 유사도 값
        all_similarities = [
            sim for matches in matched_pairs.values() 
            for _, sim, _ in matches
        ]
        
        stats = {
            'total_origin_images': len(self.origin_vectors),
            'total_generated_images': len(self.generated_vectors),
            'total_matches': total_matches,
            'unmatched_images': len(unmatched_images),
            'avg_matches_per_origin': float(avg_matches_per_origin),
            'similarity_threshold': float(self.similarity_threshold),
            'dissimilarity_threshold': float(self.dissimilarity_threshold),
            'avg_similarity': float(np.mean(all_similarities)) if all_similarities else 0.0,
            'min_similarity': float(min(all_similarities)) if all_similarities else 0.0,
            'max_similarity': float(max(all_similarities)) if all_similarities else 0.0
        }
        
        return stats

    def save_results(self):
        """매칭 결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 텍스트 형식
        with open(os.path.join(self.output_dir, 'two_stage_matching_results.txt'), 'w') as f:
            f.write(f"Two-Stage Multi-Class Image Matching Results\n")
            f.write(f"Similarity Threshold: {self.similarity_threshold:.4f}\n")
            f.write(f"Dissimilarity Threshold: {self.dissimilarity_threshold:.4f}\n")
            f.write(f"First Stage Keep Ratio: {self.first_stage_keep_ratio:.2f}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Matched Pairs\n")
            f.write("-" * 80 + "\n")
            for origin_path, matches in self.result.matched_pairs.items():
                f.write(f"Original: {origin_path}\n")
                if matches:
                    for gen_path, similarity, other_sims in matches:
                        f.write(f"  → Matched: {gen_path} (similarity: {similarity:.4f})\n")
                        for class_name, class_sim in other_sims.items():
                            f.write(f"    • {class_name}: {class_sim:.4f}\n")
                else:
                    f.write("  → No matches found above threshold\n")
                f.write("\n")
            
            f.write("\nUnmatched Images Count: {}\n".format(len(self.result.unmatched_images)))
            f.write("-" * 80 + "\n")
            f.write("First 50 unmatched images (total: {}):\n".format(len(self.result.unmatched_images)))
            for gen_path in self.result.unmatched_images[:50]:  # 처음 50개만 파일에 저장
                f.write(f"• {gen_path}\n")
        
        # 2. JSON 형식
        json_results = {
            "similarity_threshold": float(self.similarity_threshold),
            "dissimilarity_threshold": float(self.dissimilarity_threshold),
            "first_stage_keep_ratio": float(self.first_stage_keep_ratio),
            "matched_pairs": {
                origin_path: [
                    {
                        "gen_path": gen_path, 
                        "similarity": float(sim),
                        "other_class_similarities": {k: float(v) for k, v in other_sims.items()}
                    } for gen_path, sim, other_sims in matches
                ]
                for origin_path, matches in self.result.matched_pairs.items()
            },
            "unmatched_images_count": len(self.result.unmatched_images),
            "statistics": self.result.statistics
        }
        
        with open(os.path.join(self.output_dir, 'two_stage_matching_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        # 3. 매칭 통계 저장
        with open(os.path.join(self.output_dir, 'two_stage_matching_stats.txt'), 'w') as f:
            f.write(f"Two-Stage Multi-Class Matching Statistics\n")
            f.write("=" * 80 + "\n\n")
            for stat_name, stat_value in self.result.statistics.items():
                f.write(f"{stat_name}: {stat_value}\n")
                
        print(f"Results saved to {self.output_dir}")

    def copy_matched_images(self):
        """매칭된 이미지를 복사"""
        # 필터링된 이미지 저장을 위한 디렉토리 생성
        filtered_dir = os.path.join(self.output_dir, 'two_stage_filtered_images')
        rejected_dir = os.path.join(self.output_dir, 'two_stage_rejected_images')
        os.makedirs(filtered_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)
        
        # 사용된 생성 이미지를 추적하기 위한 집합
        used_generated_images = set()
        
        # 매칭된 이미지 복사
        print("\nCopying matched images...")
        count = 0
        for origin_path, matches in self.result.matched_pairs.items():
            for gen_path, similarity, other_class_sims in matches:
                # 원본 파일의 전체 경로
                src_path = os.path.join(self.generated_dir, gen_path)
                
                # 원래 파일명 유지하고 유사도 추가
                filename = os.path.basename(gen_path)
                name, ext = os.path.splitext(filename)
                
                # 다른 클래스와의 최대 유사도 계산
                max_other_sim = max(other_class_sims.values()) if other_class_sims else 0
                
                # 파일명에 원본 유사도와 다른 클래스 최대 유사도 포함
                new_name = f"{name}_sim{similarity:.4f}_other{max_other_sim:.4f}{ext}"
                
                # 대상 파일의 전체 경로
                dst_path = os.path.join(filtered_dir, new_name)
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    used_generated_images.add(gen_path)
                    count += 1
                    # 100개 단위로 진행 상황만 출력
                    if count % 100 == 0:
                        print(f"  Copied {count} matched images so far...")
                else:
                    if VERBOSE:
                        print(f"Warning: Source file not found: {src_path}")
        
        # 매칭되지 않은 이미지 모두 복사 
        print("\nCopying ALL rejected images...")
        count = 0
        total_unmatched = len(self.result.unmatched_images)
        
        # tqdm으로 진행 상황 표시
        for gen_path in tqdm(self.result.unmatched_images, total=total_unmatched, desc="Copying rejected"):
            # 원본 파일의 전체 경로
            src_path = os.path.join(self.generated_dir, gen_path)
            
            # 원래 파일명 유지
            filename = os.path.basename(gen_path)
            
            # 대상 파일의 전체 경로
            dst_path = os.path.join(rejected_dir, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                count += 1
                # 1000개 단위로 진행 상황 출력
                if count % 1000 == 0:
                    print(f"  Copied {count}/{total_unmatched} rejected images...")
            else:
                if VERBOSE:
                    print(f"Warning: Source file not found: {src_path}")
            
        print(f"\n=== Processing Completed ===")
        print(f"Filtered images saved to: {filtered_dir}")
        print(f"All {count} rejected images saved to: {rejected_dir}")
        print(f"Total original images: {self.result.statistics['total_origin_images']}")
        print(f"Total generated images: {self.result.statistics['total_generated_images']}")
        print(f"Total matches: {self.result.statistics['total_matches']}")
        print(f"Unmatched images: {self.result.statistics['unmatched_images']}")
        print(f"Average matches per original image: {self.result.statistics['avg_matches_per_origin']:.2f}")
        print(f"Average similarity: {self.result.statistics['avg_similarity']:.4f}")


def match_keratitis_two_stage(similarity_threshold=0.7, 
                              dissimilarity_threshold=0.5, 
                              first_stage_keep_ratio=0.3):
    """Keratitis 이미지 다중 클래스 매칭 (2단계 방식)"""
    print("\n=== Matching Keratitis Images with Two-Stage Filtering ===")
    
    # 경로 설정
    vector_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/vectors"
    origin_vector_path = os.path.join(vector_dir, "keratitis.npy")
    generated_vector_path = os.path.join(vector_dir, "generated_keratitis.npy")
    
    # 다른 클래스 벡터 경로 설정
    other_class_vector_paths = {
        # "normal": os.path.join(vector_dir, "normal.npy"),
        "각막궤양": os.path.join(vector_dir, "각막궤양.npy"),
        # "각막부골편": os.path.join(vector_dir, "각막부골편.npy"),
        "결막염": os.path.join(vector_dir, "결막염.npy"),
        # "blepharitis": os.path.join(vector_dir, "blepharitis.npy"),
        # "generated_blepharitis": os.path.join(vector_dir, "generated_blepharitis.npy")
    }
    
    generated_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/generated/keratitis"
    output_dir = "/home/minelab/desktop/Jack/step_vet_train/datasets/two_stage_filtered/keratitis"
    
    # 2단계 매칭 실행
    TwoStageImageMatcher(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        other_class_vector_paths=other_class_vector_paths,
        generated_dir=generated_dir,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
        dissimilarity_threshold=dissimilarity_threshold,
        matches_per_image=1,
        first_stage_keep_ratio=first_stage_keep_ratio
    )


if __name__ == "__main__":
    # 매칭 실행 (유사도 임계값 0.7, 비유사도 임계값 0.5로 설정, 1단계에서 상위 30% 유지)
    similarity_threshold = 0.7
    dissimilarity_threshold = 0.5
    first_stage_keep_ratio = 0.3  # 상위 30%만 유지
    
    try:
        match_keratitis_two_stage(
            similarity_threshold=similarity_threshold,
            dissimilarity_threshold=dissimilarity_threshold,
            first_stage_keep_ratio=first_stage_keep_ratio
        )
    except Exception as e:
        print(f"Keratitis 2단계 매칭 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 2단계 매칭 프로세스 완료 ===")
