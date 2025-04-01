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
   - Step for fine-tuning the encoder model
   - Execute these scripts in the `finetuning_encoder/` directory:
   ```bash
   cd finetuning_encoder
   python prepare_data.py  # Prepare data for encoder fine-tuning
   python train.py  # Run the encoder fine-tuning process
   python evaluate.py  # Evaluate the fine-tuned encoder
   ```

3. **Query Strategy**
   - Final step for implementing query strategies
   - Execute these scripts in the `query_strategy/` directory:
   ```bash
   cd query_strategy
   python active_learning.py  # Run active learning query strategy
   python run.py  # Execute the main query processing
   python evaluate_results.py  # Evaluate your query strategy results
   ```

Each step must be executed sequentially, as the output from previous steps serves as input for subsequent steps.

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