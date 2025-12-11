#!/bin/bash

# Полный пайплайн: разметка -> конвертация -> обучение
# Используйте этот скрипт как шаблон

# ====== НАСТРОЙКИ ======
IMAGES_DIR="path/to/your/images"
ANNOTATIONS_JSON="vlm_annotations.json"
LLAVA_TRAIN="llava_train.json"
LLAVA_VAL="llava_val.json"
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

# ====== ШАГ 1: РАЗМЕТКА ======
echo "Step 1: Annotating images..."
python vlm_annotation/data_annotation.py \
    --images_dir "$IMAGES_DIR" \
    --output_file "$ANNOTATIONS_JSON" \
    --yolo_model "$YOLO_MODEL" \
    --detr_model "$DETR_MODEL" \
    --detr_processor "$DETR_PROCESSOR" \
    --llm_model "Salesforce/blip-image-captioning-large" \
    --conf_threshold 0.3

if [ $? -ne 0 ]; then
    echo "Error in annotation step!"
    exit 1
fi

# ====== ШАГ 2: РАЗДЕЛЕНИЕ НА TRAIN/VAL ======
echo "Step 2: Splitting data into train/val..."
python - << EOF
import json
import random
from pathlib import Path

random.seed(42)

with open("$ANNOTATIONS_JSON", 'r') as f:
    data = json.load(f)

random.shuffle(data)

split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
val_data = data[split_idx:]

with open("$LLAVA_TRAIN", 'w') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open("$LLAVA_VAL", 'w') as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")
EOF

# ====== ШАГ 3: КОНВЕРТАЦИЯ В ФОРМАТ LLaVA ======
echo "Step 3: Converting to LLaVA format..."

python vlm_training/convert_to_llava_format.py \
    --annotation_file "$LLAVA_TRAIN" \
    --output_file "${LLAVA_TRAIN%.json}_llava.json"

python vlm_training/convert_to_llava_format.py \
    --annotation_file "$LLAVA_VAL" \
    --output_file "${LLAVA_VAL%.json}_llava.json"

if [ $? -ne 0 ]; then
    echo "Error in conversion step!"
    exit 1
fi

# ====== ШАГ 4: ОБУЧЕНИЕ VLM ======
echo "Step 4: Training VLM model..."
python vlm_training/train_vlm.py \
    --train_data "${LLAVA_TRAIN%.json}_llava.json" \
    --val_data "${LLAVA_VAL%.json}_llava.json" \
    --output_dir "$VLM_OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --image_dir "$IMAGES_DIR" \
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

