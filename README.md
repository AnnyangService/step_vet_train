# step_vet

## Quick Start Guide

This project must be executed in the following sequential order:

(임시라서 제대로 새로 작성해야 함.)

1. **Generation**
   - First step for data generation
   - Execute these scripts in the `generation/datapreparation` 실행, 생성할 데이터 아래와 같이 파라미터 설정
   ```
   disease_mapping = {
      '안검염': 'blepharitis',
      '비궤양성각막염': 'keratitis'
    }
    ```
   - 2. generation/train.py 실행, 체크포인트에서 오류나면 100번째 줄 경로 변경하기
   - 3. generation/generate.py 실행
    

2. **Finetuning Encoder**
   1. finetuning_encoder/gt_mask_generation/mask_generation_train.py을 마스크 생성 모델 학습
   2. finetuning_encoder/gt_mask_generation/mask_generation.py를 실행해서 마스크 생성
   3. finetuning_encoder/gt_mask_generation/mask_classification.py를 실행해서 마스크 분류
   4. finetuning_encoder/finetuning.py를 실행해서 인코더 파인튜닝
   5. visuialization/before_channel_selection.py로 인코더 파인튜닝 결과 확인 가능

3. **Query Strategy**
   - 제일 먼저 벡터화 query_strategy/encoder/vectorization.py 실행해서 이미지들 벡터화, 97, 98 line을 보면 input, output을 설정할 수 있는데 input 대상은 generated_keratitis, keratitis, generated_blepharitis, blepharitis 이런식으로 지정하면 됨.
   - 다음은 query_staretegy/filtering/image_matching_filtering.py 혹은 query_staretegy/filtering/thresholding_filtering.py, 앞에꺼는 이미지 하나당 유사한 두 개의 이미지 뽑는 방법, 뒤에꺼는 원본 이미지의 평균 벡터와 모든 생성된 이미지 비교해서 코사인 유사도의 평균을 잡고, 그거보다 낮은 이미지 제거하는 방법

4. **진단 모델 학습 - YOLO**
   - YOLO 학습을 위한 데이터셋 정제
      1. utils/refine_rare_disease.py로 필터링된 이미지를 가지고 rare disease 3100장으로 맞춤
      2. utils/refine_normal_disease.py 로 normal 데이터셋 필요한 개수만큼 정제(일반 3100, anmaly detection 기법 사용 시 9300)
      3. utils/refine_other_disease.py로 그 외 질병들 3100장으로 맞춤
      4. split_dataset_yolo.py로 train/val/test를 yolo 데이터 폼으로 맞추는데, val, test는 생성된 이미지가 포함되지 않아야 함. 그래서 생성해서 모두 3100장으로 맞추긴 했지만, 3100장으로 코드를 돌려버리면 train용 blepharitis 질병은 거의 모두 생성된 이미지만 들어있어서 3100보다는 낮은 값 추천 디폴트는 1500
      5. diagnosis/yolo_v11/train.py 로 학습, diagnosis/yolo_v11/evaluate.py로 평가

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for training)
- Git

### Setting Up the Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/step_vet.git
cd step_vet

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for visualization (optional)
pip install matplotlib seaborn
```

### Data Requirements
- Place your raw data in the `datasets/` directory
- Ensure image data is in the correct format (jpg/png)
- Annotation files should be in the appropriate format as specified in the documentation

### Configuration
- Modify configuration parameters in each script as needed
- Adjust hyperparameters in respective config files before running each step

## Troubleshooting
- If you encounter CUDA errors, check your GPU compatibility and driver versions
- For memory issues, reduce batch size in the configuration files
- See the documentation in each directory for specific debugging information