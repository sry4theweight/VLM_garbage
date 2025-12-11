# Руководство по использованию VLM системы

## Обзор

Эта система позволяет создавать Visual Language Model (VLM), которая может:
1. Описывать объекты на изображениях с учетом их расположения (например, "here is a glass on the grass")
2. Отвечать на вопросы по изображениям (VQA - Visual Question Answering)
3. Работать с координатами bounding boxes для описания конкретных объектов

## Архитектура решения

### Этап 1: Разметка данных

1. **Ансамбль детекции** (YOLO + RT-DETR) находит объекты на изображениях и возвращает координаты bbox
2. **LLM** (BLIP) генерирует текстовые описания для каждого объекта с учетом контекста
3. Автоматически создаются Q&A пары для обучения VQA

### Этап 2: Обучение VLM

Fine-tuning модели LLaVA на размеченных данных с использованием LoRA для эффективного обучения.

## Рекомендуемый workflow

### 1. Подготовка данных

```bash
# Соберите изображения в одну директорию
mkdir -p data/images
# Поместите ваши изображения в data/images/
```

### 2. Разметка

```bash
python vlm_annotation/data_annotation.py \
    --images_dir data/images \
    --output_file annotations.json \
    --yolo_model models/yolo/yolov8x/best.pt \
    --detr_model models/rt-detr/rt-detr-101/m \
    --detr_processor models/rt-detr/rt-detr-101/p \
    --conf_threshold 0.3
```

**Что происходит:**
- Модели детекции находят все объекты на каждом изображении
- Для каждого объекта генерируется описание с использованием LLM
- Создаются вопросы и ответы для обучения VQA

### 3. Разделение данных

```bash
python vlm_training/split_data.py \
    --input_file annotations.json \
    --train_file train.json \
    --val_file val.json \
    --train_ratio 0.8 \
    --val_ratio 0.1
```

### 4. Конвертация в формат LLaVA

```bash
python vlm_training/convert_to_llava_format.py \
    --annotation_file train.json \
    --output_file train_llava.json

python vlm_training/convert_to_llava_format.py \
    --annotation_file val.json \
    --output_file val_llava.json
```

### 5. Обучение

```bash
python vlm_training/train_vlm.py \
    --train_data train_llava.json \
    --val_data val_llava.json \
    --output_dir ./vlm_model \
    --base_model llava-hf/llava-1.5-7b-hf \
    --image_dir data/images \
    --use_lora \
    --batch_size 4 \
    --num_epochs 3
```

### 6. Использование

```python
from vlm_inference.vlm_inference import VLMInference

vlm = VLMInference(
    model_path="./vlm_model",
    base_model="llava-hf/llava-1.5-7b-hf"
)

# Описание объекта
description = vlm.describe_object(
    "image.jpg",
    bbox=[150, 200, 300, 400],
    label="glass"
)
print(description)  # "Here is a glass. The glass is on the grass in a park."

# Ответ на вопрос
answer = vlm.answer_question(
    "image.jpg",
    "How many objects are there?"
)
print(answer)  # "There are 3 objects in this image."
```

## Альтернативные подходы

### Использование более легких моделей

Если у вас ограниченные ресурсы GPU:

1. **Использование quantized моделей:**
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
```

2. **Использование меньших моделей:**
- `llava-hf/llava-1.5-7b-hf` вместо 13B версии
- Или использование более легких моделей типа BLIP-2

### Улучшение качества описаний

Для получения более точных описаний можно:

1. **Использовать более мощные LLM для разметки:**
   - `Salesforce/blip2-opt-2.7b` - более мощная модель для генерации описаний
   - Или использовать API (GPT-4 Vision, Claude Vision)

2. **Улучшить промпты:**
   - Редактировать функцию `generate_description` в `data_annotation.py`
   - Добавить более специфичные промпты для разных классов объектов

### Оптимизация обучения

1. **Gradient checkpointing:**
```python
training_args.gradient_checkpointing = True
```

2. **Уменьшение размера батча** если не хватает памяти

3. **Использование mixed precision training** (уже включено с fp16)

## Troubleshooting

### Проблема: Out of memory при обучении

**Решения:**
- Уменьшите `batch_size` (до 1-2)
- Используйте gradient accumulation
- Уменьшите размер изображений
- Используйте quantization

### Проблема: Модель генерирует неправильные описания

**Решения:**
- Увеличьте количество эпох обучения
- Проверьте качество разметки
- Добавьте больше разнообразных примеров в датасет
- Настройте learning rate

### Проблема: Медленная разметка

**Решения:**
- Используйте GPU для LLM
- Обрабатывайте изображения батчами (требует модификации кода)
- Используйте более легкую LLM модель для разметки

## Дополнительные возможности

### Добавление пользовательских промптов

Вы можете модифицировать `data_annotation.py` для добавления специфичных промптов для разных классов:

```python
def generate_description(self, image, label, bbox):
    # Специфичные промпты для разных классов
    prompts = {
        'glass': "a glass bottle. Describe what surface it is on and what is around it.",
        'plastic': "a plastic object. Describe its location and surroundings.",
        # ...
    }
    prompt = prompts.get(label, f"a {label}. Describe where it is.")
    # ...
```

### Добавление больше Q&A пар

Модифицируйте функцию `generate_qa_pairs` для добавления большего количества вопросов и ответов.

## Следующие шаги

1. Соберите больше данных для лучшего качества модели
2. Экспериментируйте с гиперпараметрами обучения
3. Оцените качество модели на тестовом наборе
4. Оптимизируйте модель для production использования

