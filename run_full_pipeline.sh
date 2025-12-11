#!/bin/bash

# Полный пайплайн: скачивание датасета -> разметка -> конвертация -> обучение
# Используйте этот скрипт как шаблон

# ====== НАСТРОЙКИ ======
# Roboflow настройки
ROBOFLOW_API_KEY="${ROBOFLOW_API_KEY:-}"  # Можно установить через export ROBOFLOW_API_KEY="your-key"
ROBOFLOW_WORKSPACE="kuivashev"
ROBOFLOW_PROJECT="my-normal-dataset"
ROBOFLOW_VERSION=1
ROBOFLOW_OUTPUT_DIR="data/roboflow_dataset"

# Пути для разметки
VLM_ANNOTATIONS_DIR="data/vlm_annotations"
ANNOTATIONS_JSON="data/vlm_annotations/all_annotations.json"
LLAVA_TRAIN="data/llava_train.json"
LLAVA_VAL="data/llava_val.json"
VLM_OUTPUT_DIR="./vlm_model"
BASE_MODEL="llava-hf/llava-1.5-7b-hf"

# Модели детекции
YOLO_MODEL="models/yolo/yolov8x/best.pt"
DETR_MODEL="models/rt-detr/rt-detr-101/m"
DETR_PROCESSOR="models/rt-detr/rt-detr-101/p"

# Параметры обучения
BATCH_SIZE=4
LEARNING_RATE=2e-5
NUM_EPOCHS=3

# ====== ШАГ 0: СКАЧИВАНИЕ ДАТАСЕТА ROBOFLOW ======
echo "Step 0: Downloading Roboflow dataset..."
python download_roboflow_dataset.py \
    --api_key "$ROBOFLOW_API_KEY" \
    --workspace "$ROBOFLOW_WORKSPACE" \
    --project "$ROBOFLOW_PROJECT" \
    --version "$ROBOFLOW_VERSION" \
    --output_dir "$ROBOFLOW_OUTPUT_DIR" \
    --format "coco"

if [ $? -ne 0 ]; then
    echo "Error downloading dataset!"
    exit 1
fi

# Находим путь к скачанному датасету
DATASET_PATH=$(find "$ROBOFLOW_OUTPUT_DIR" -type d -name "${ROBOFLOW_PROJECT}-${ROBOFLOW_VERSION}" | head -n 1)
if [ -z "$DATASET_PATH" ]; then
    echo "Error: Could not find downloaded dataset directory"
    exit 1
fi

echo "Dataset downloaded to: $DATASET_PATH"

# ====== ШАГ 1: РАЗМЕТКА ======
echo "Step 1: Annotating Roboflow dataset..."
python annotate_roboflow_dataset.py \
    --roboflow_dataset_dir "$DATASET_PATH" \
    --output_dir "$VLM_ANNOTATIONS_DIR" \
    --yolo_model "$YOLO_MODEL" \
    --detr_model "$DETR_MODEL" \
    --detr_processor "$DETR_PROCESSOR" \
    --llm_model "Salesforce/blip-image-captioning-large" \
    --conf_threshold 0.3 \
    --splits train valid test

if [ $? -ne 0 ]; then
    echo "Error in annotation step!"
    exit 1
fi

# Объединяем аннотации из всех сплитов
echo "Step 1.5: Combining annotations from all splits..."
python3 << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

annotations_dir = Path(sys.argv[1])
output_file = Path(sys.argv[2])

all_data = []

for split_file in ["train_annotations.json", "valid_annotations.json", "test_annotations.json"]:
    file_path = annotations_dir / split_file
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
        print(f"Loaded {len(data)} samples from {split_file}")

output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"Total: {len(all_data)} samples combined")
print(f"Saved to: {output_file}")
PYTHON_SCRIPT "$VLM_ANNOTATIONS_DIR" "$ANNOTATIONS_JSON"

# ====== ШАГ 2: РАЗДЕЛЕНИЕ НА TRAIN/VAL ======
echo "Step 2: Splitting data into train/val..."
python vlm_training/split_data.py \
    --input_file "$ANNOTATIONS_JSON" \
    --train_file "${LLAVA_TRAIN%.json}.json" \
    --val_file "${LLAVA_VAL%.json}.json" \
    --train_ratio 0.8 \
    --val_ratio 0.2

# ====== ШАГ 3: КОНВЕРТАЦИЯ В ФОРМАТ LLaVA ======
echo "Step 3: Converting to LLaVA format..."

python vlm_training/convert_to_llava_format.py \
    --annotation_file "${LLAVA_TRAIN%.json}.json" \
    --output_file "$LLAVA_TRAIN"

python vlm_training/convert_to_llava_format.py \
    --annotation_file "${LLAVA_VAL%.json}.json" \
    --output_file "$LLAVA_VAL"

if [ $? -ne 0 ]; then
    echo "Error in conversion step!"
    exit 1
fi

# ====== ШАГ 4: ОБУЧЕНИЕ VLM ======
echo "Step 4: Training VLM model..."
python vlm_training/train_vlm.py \
    --train_data "$LLAVA_TRAIN" \
    --val_data "$LLAVA_VAL" \
    --output_dir "$VLM_OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --image_dir "$DATASET_PATH" \
    --use_lora \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS"

if [ $? -ne 0 ]; then
    echo "Error in training step!"
    exit 1
fi

echo "Pipeline completed successfully!"
echo "Model saved to: $VLM_OUTPUT_DIR"

