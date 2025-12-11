#!/bin/bash

# Пример скрипта для разметки изображений
# Используйте этот скрипт как шаблон и настройте пути под вашу систему

IMAGES_DIR="path/to/your/images"
OUTPUT_FILE="vlm_annotations.json"
YOLO_MODEL="models/yolo/yolov8x/best.pt"
DETR_MODEL="models/rt-detr/rt-detr-101/m"
DETR_PROCESSOR="models/rt-detr/rt-detr-101/p"

python vlm_annotation/data_annotation.py \
    --images_dir "$IMAGES_DIR" \
    --output_file "$OUTPUT_FILE" \
    --yolo_model "$YOLO_MODEL" \
    --detr_model "$DETR_MODEL" \
    --detr_processor "$DETR_PROCESSOR" \
    --llm_model "Salesforce/blip-image-captioning-large" \
    --conf_threshold 0.3

echo "Annotation completed! Output saved to $OUTPUT_FILE"

