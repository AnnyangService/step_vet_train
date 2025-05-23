import os
import shutil
import random
from pathlib import Path
from collections import Counter

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def get_random_files(directory, num_files, extensions=(".jpg", ".png")):
    files = [f for f in os.listdir(directory) if f.endswith(extensions)]
    if len(files) < num_files:
        raise ValueError(f"Not enough files in {directory}. Required: {num_files}, Available: {len(files)}")
    return random.sample(files, num_files)

def copy_files(source_files, source_dir, target_dir, class_name):
    for file in source_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(target_dir, class_name, file)
        shutil.copy2(src, dst)

def main():
    random.seed(42)

    # 각 질병별 사용할 이미지 수를 직접 지정
    disease_image_counts = {
        'blepharitis': 1500,
        'keratitis': 1500,
        'conjunctivitis': 1500,
        '각막궤양': 2250,
        '각막부골편': 2250
    }
    val_count = 150
    test_count = 150

    # inflammatory/corneal_disease 클래스별 총 이미지 수 자동 계산
    num_images = {
        'inflammatory': disease_image_counts['blepharitis'] + disease_image_counts['keratitis'] + disease_image_counts['conjunctivitis'],
        'corneal_disease': disease_image_counts['각막궤양'] + disease_image_counts['각막부골편']
    }

    # Base directories
    base_dir = Path("/home/minelab/desktop/Jack/step_vet_train/datasets")
    origin_dir = base_dir / "origin"
    filtered_blepharitis_dir = base_dir / "matching_filtered/blepharitis/filtered_images"
    filtered_keratitis_dir = base_dir / "matching_filtered/keratitis/filtered_images"

    output_base = base_dir / "step2"
    splits = ['train', 'val', 'test']
    classes = ['inflammatory', 'corneal_disease']

    for split in splits:
        for class_name in classes:
            create_directory(output_base / split / class_name)

    # inflammatory: blepharitis + keratitis + conjunctivitis
    blepharitis_dir = origin_dir / 'blepharitis'
    keratitis_dir = origin_dir / 'keratitis'
    conjunctivitis_dir = origin_dir / '결막염'
    blepharitis_files = [f for f in os.listdir(blepharitis_dir) if f.endswith('.jpg') or f.endswith('.png')]
    keratitis_files = [f for f in os.listdir(keratitis_dir) if f.endswith('.jpg') or f.endswith('.png')]
    conjunctivitis_files = [f for f in os.listdir(conjunctivitis_dir) if f.endswith('.jpg') or f.endswith('.png')]
    filtered_blepharitis_files = [f for f in os.listdir(filtered_blepharitis_dir) if f.endswith('.jpg') or f.endswith('.png')]
    filtered_keratitis_files = [f for f in os.listdir(filtered_keratitis_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # blepharitis
    needed_blepharitis = max(0, disease_image_counts['blepharitis'] - len(blepharitis_files))
    if needed_blepharitis > 0:
        blepharitis_files.extend(get_random_files(filtered_blepharitis_dir, needed_blepharitis))
    else:
        blepharitis_files = random.sample(blepharitis_files, disease_image_counts['blepharitis'])
    # keratitis
    needed_keratitis = max(0, disease_image_counts['keratitis'] - len(keratitis_files))
    if needed_keratitis > 0:
        keratitis_files.extend(get_random_files(filtered_keratitis_dir, needed_keratitis))
    else:
        keratitis_files = random.sample(keratitis_files, disease_image_counts['keratitis'])
    # conjunctivitis
    if len(conjunctivitis_files) < disease_image_counts['conjunctivitis']:
        raise ValueError("결막염 이미지가 부족합니다.")
    else:
        conjunctivitis_files = random.sample(conjunctivitis_files, disease_image_counts['conjunctivitis'])
    inflammatory_files = [("blepharitis", f) for f in blepharitis_files] + [("keratitis", f) for f in keratitis_files] + [("conjunctivitis", f) for f in conjunctivitis_files]
    random.shuffle(inflammatory_files)
    # seed 파일 분리 (blepharitis, keratitis 모두)
    seed_files = [(src, f) for src, f in inflammatory_files if (src == 'blepharitis' and f.startswith('seed')) or (src == 'keratitis' and f.startswith('seed'))]
    non_seed_files = [(src, f) for src, f in inflammatory_files if not ((src == 'blepharitis' and f.startswith('seed')) or (src == 'keratitis' and f.startswith('seed')))]

    # corneal disease: 각막궤양 + 각막부골편
    ulcer_dir = origin_dir / '각막궤양'
    fb_dir = origin_dir / '각막부골편'
    ulcer_files = [f for f in os.listdir(ulcer_dir) if f.endswith('.jpg') or f.endswith('.png')]
    fb_files = [f for f in os.listdir(fb_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if len(ulcer_files) < disease_image_counts['각막궤양']:
        raise ValueError("각막궤양 이미지가 부족합니다.")
    if len(fb_files) < disease_image_counts['각막부골편']:
        raise ValueError("각막부골편 이미지가 부족합니다.")
    ulcer_files = random.sample(ulcer_files, disease_image_counts['각막궤양'])
    fb_files = random.sample(fb_files, disease_image_counts['각막부골편'])
    corneal_files = [("각막궤양", f) for f in ulcer_files] + [("각막부골편", f) for f in fb_files]
    random.shuffle(corneal_files)

    # Split
    def split_files(files, val_count, test_count, seed_check=None):
        if seed_check is not None:
            seed = [x for x in files if seed_check(x)]
            non_seed = [x for x in files if not seed_check(x)]
            assert all(not seed_check(x) for x in non_seed[:val_count+test_count])
            train = seed + non_seed[:len(files)-val_count-test_count-len(seed)]
            val = non_seed[len(files)-val_count-test_count-len(seed):len(files)-test_count-len(seed)]
            test = non_seed[len(files)-test_count-len(seed):]
            return train, val, test
        else:
            train = files[:len(files)-val_count-test_count]
            val = files[len(files)-val_count-test_count:len(files)-test_count]
            test = files[len(files)-test_count:]
            return train, val, test

    train_inflam, val_inflam, test_inflam = split_files(
        seed_files + non_seed_files, val_count, test_count,
        seed_check=lambda x: (x[0]=='blepharitis' and x[1].startswith('seed')) or (x[0]=='keratitis' and x[1].startswith('seed'))
    )
    train_corneal, val_corneal, test_corneal = split_files(corneal_files, val_count, test_count)

    split_map = {
        'train': {
            'inflammatory': train_inflam,
            'corneal_disease': train_corneal
        },
        'val': {
            'inflammatory': val_inflam,
            'corneal_disease': val_corneal
        },
        'test': {
            'inflammatory': test_inflam,
            'corneal_disease': test_corneal
        }
    }

    for split in splits:
        for class_name, files in split_map[split].items():
            if class_name == 'inflammatory':
                for src, file in files:
                    if src == 'blepharitis' and file.startswith('seed'):
                        src_dir = filtered_blepharitis_dir
                    elif src == 'keratitis' and file.startswith('seed'):
                        src_dir = filtered_keratitis_dir
                    elif src == 'blepharitis':
                        src_dir = blepharitis_dir
                    elif src == 'keratitis':
                        src_dir = keratitis_dir
                    else:  # conjunctivitis
                        src_dir = conjunctivitis_dir
                    dst = os.path.join(output_base, split, class_name, file)
                    shutil.copy2(os.path.join(src_dir, file), dst)
            else:  # corneal_disease
                for src, file in files:
                    src_dir = origin_dir / src
                    dst = os.path.join(output_base, split, class_name, file)
                    shutil.copy2(os.path.join(src_dir, file), dst)

    # === 세부 클래스별 포함 개수 출력 ===
    print("\n[Split별 세부 클래스별 이미지 개수]")
    for split in splits:
        print(f"\n[{split.upper()}]")
        for class_name in classes:
            files = split_map[split][class_name]
            if class_name == 'inflammatory':
                sub_counts = Counter([src for src, _ in files])
                print(f"  {class_name}: {sum(sub_counts.values())}개", end='  ')
                print(', '.join([f"{k}: {v}" for k, v in sub_counts.items()]))
            else:
                sub_counts = Counter([src for src, _ in files])
                print(f"  {class_name}: {sum(sub_counts.values())}개", end='  ')
                print(', '.join([f"{k}: {v}" for k, v in sub_counts.items()]))

if __name__ == "__main__":
    main() 