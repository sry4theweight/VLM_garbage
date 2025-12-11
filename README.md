# Система распознавания мусора с VLM

Проект включает систему детекции объектов мусора с использованием ансамбля моделей (YOLO + RT-DETR) и Visual Language Model (VLM) для описания объектов и ответов на вопросы по изображениям.

## Быстрый старт

### Обновление проекта из Git

```bash
git pull origin main
```

### Обновление проекта в Git

```bash
git add . && git commit -m "Update VLM pipeline" && git push origin main
```

Подробные инструкции по Git: см. [git_commands.md](git_commands.md)

### Скачивание и разметка датасета Roboflow

```bash
# 1. Скачать датасет (по умолчанию в формате COCO)
python download_roboflow_dataset.py

# Или с явным указанием параметров:
python download_roboflow_dataset.py \
    --api_key "IYwUdRPmNzuSjy8A6cOr" \
    --workspace "kuivashev" \
    --project "complete-wxatb" \
    --version 1 \
    --output_dir "data/roboflow_dataset" \
    --format "coco"

# 2. Разметить датасет
python annotate_roboflow_dataset.py \
    --roboflow_dataset_dir "data/roboflow_dataset" \
    --output_dir "data/vlm_annotations"
```

Или используйте полный пайплайн: `./run_full_pipeline.sh`

## Структура проекта

```
.
├── models/                      # Модели детекции объектов
│   ├── yolo/                   # Модели YOLO
│   │   ├── yolov8x/
│   │   ├── yolov8l/
│   │   └── yolov8m/
│   └── rt-detr/                # Модели RT-DETR
│       ├── rt-detr-50/
│       ├── rt-detr-101/
│       └── rt-detr-101-better-glass/
├── vlm_annotation/             # Скрипты для разметки данных
│   ├── ensemble_detector.py    # Ансамбль моделей детекции
│   └── data_annotation.py      # Разметка изображений с LLM
├── vlm_training/               # Скрипты для обучения VLM
│   ├── convert_to_llava_format.py  # Конвертация в формат LLaVA
│   └── train_vlm.py            # Fine-tuning VLM модели
├── vlm_inference/              # Инференс обученной VLM
│   └── vlm_inference.py        # Скрипт для использования VLM
├── download_roboflow_dataset.py # Скачивание датасета Roboflow
├── annotate_roboflow_dataset.py # Разметка датасета Roboflow
├── server.ipynb                # Сервер для обработки видео
├── run_full_pipeline.sh        # Полный пайплайн (скачивание -> разметка -> обучение)
├── git_commands.md             # Инструкции по работе с Git
└── training_and_testing_steps/ # Обучение и тестирование моделей детекции
```

## Установка

### Вариант 1: Через Conda (рекомендуется)

```bash
# Создание окружения из environment.yml
conda env create -f environment.yml

# Активация окружения
conda activate vlm_garbage

# Проверка GPU
python check_gpu.py
```

### Вариант 2: Через pip

```bash
# Установка PyTorch с CUDA (выберите версию CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Установка остальных зависимостей
pip install -r requirements_vlm.txt

# Проверка GPU
python check_gpu.py
```

### Модели детекции

Убедитесь, что у вас есть модели детекции в директории `models/`:
- YOLO модели: `models/yolo/yolov8x/best.pt`
- RT-DETR модели: `models/rt-detr/rt-detr-101/m/` и `models/rt-detr/rt-detr-101/p/`

## Использование

### 0. Скачивание датасета Roboflow

Для скачивания датасета из Roboflow:

```bash
# Простой вариант (API ключ уже прописан в скрипте):
python download_roboflow_dataset.py

# Или с явным указанием параметров:
python download_roboflow_dataset.py \
    --api_key "IYwUdRPmNzuSjy8A6cOr" \
    --workspace "kuivashev" \
    --project "complete-wxatb" \
    --version 1 \
    --output_dir "data/roboflow_dataset" \
    --format "coco"
```

Или используйте полный пайплайн (см. ниже).

### 1. Разметка данных для обучения VLM

#### Для датасета Roboflow (рекомендуется):

```bash
python annotate_roboflow_dataset.py \
    --roboflow_dataset_dir "data/roboflow_dataset" \
    --output_dir "data/vlm_annotations" \
    --yolo_model "models/yolo/yolov8x/best.pt" \
    --detr_model "models/rt-detr/rt-detr-101/m" \
    --detr_processor "models/rt-detr/rt-detr-101/p"
```

#### Для произвольных изображений:

Разметка изображений с использованием ансамбля моделей детекции и LLM:

```bash
python vlm_annotation/data_annotation.py \
    --images_dir path/to/images \
    --output_file annotations.json \
    --yolo_model models/yolo/yolov8x/best.pt \
    --detr_model models/rt-detr/rt-detr-101/m \
    --detr_processor models/rt-detr/rt-detr-101/p \
    --llm_model Salesforce/blip-image-captioning-large \
    --conf_threshold 0.3
```

**Параметры:**
- `--images_dir`: директория с изображениями для разметки
- `--output_file`: путь к выходному JSON файлу с разметкой
- `--yolo_model`: путь к модели YOLO
- `--detr_model`: путь к модели RT-DETR (директория m/)
- `--detr_processor`: путь к процессору RT-DETR (директория p/)
- `--llm_model`: модель LLM из HuggingFace для генерации описаний
- `--conf_threshold`: порог уверенности для детекций (по умолчанию 0.3)

**Формат выходного файла:**
```json
[
  {
    "image": "path/to/image.jpg",
    "detections": [
      {
        "bbox": [x1, y1, x2, y2],
        "label": "glass",
        "confidence": 0.95,
        "description": "Here is a glass. The glass is on the grass in a park."
      }
    ],
    "qa_pairs": [
      {
        "question": "How many objects are there?",
        "answer": "There are 3 objects in this image."
      },
      {
        "question": "Is there any glass in this image?",
        "answer": "Yes, there is 1 glass object."
      }
    ]
  }
]
```

### 2. Разделение данных на train/val (опционально)

Если нужно разделить данные на train/val/test:

```bash
python vlm_training/split_data.py \
    --input_file annotations.json \
    --train_file train_annotations.json \
    --val_file val_annotations.json \
    --test_file test_annotations.json \
    --train_ratio 0.8 \
    --val_ratio 0.1
```

### 3. Конвертация в формат LLaVA

Конвертация разметки в формат для обучения LLaVA:

```bash
python vlm_training/convert_to_llava_format.py \
    --annotation_file train_annotations.json \
    --output_file llava_train.json

python vlm_training/convert_to_llava_format.py \
    --annotation_file val_annotations.json \
    --output_file llava_val.json
```

### 4. Обучение VLM

Fine-tuning модели LLaVA на размеченных данных:

```bash
python vlm_training/train_vlm.py \
    --train_data llava_train.json \
    --val_data llava_val.json \
    --output_dir ./vlm_model \
    --base_model llava-hf/llava-1.5-7b-hf \
    --image_dir path/to/images \
    --use_lora \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

**Параметры:**
- `--train_data`: JSON файл с обучающими данными (формат LLaVA)
- `--val_data`: JSON файл с валидационными данными (опционально)
- `--output_dir`: директория для сохранения обученной модели
- `--base_model`: базовая модель LLaVA из HuggingFace
- `--image_dir`: базовый путь к директории с изображениями
- `--use_lora`: использовать LoRA для эффективного fine-tuning
- `--batch_size`: размер батча (по умолчанию 4)
- `--learning_rate`: скорость обучения (по умолчанию 2e-5)
- `--num_epochs`: количество эпох (по умолчанию 3)

**Рекомендуемые модели:**
- `llava-hf/llava-1.5-7b-hf` - 7B параметров (требует ~14GB VRAM)
- `llava-hf/llava-1.5-13b-hf` - 13B параметров (требует ~26GB VRAM)

Для меньших ресурсов можно использовать QLoRA или quantized модели.

### 5. Использование обученной VLM

Ответ на вопросы по изображению:

```bash
python vlm_inference/vlm_inference.py \
    --model_path ./vlm_model \
    --base_model llava-hf/llava-1.5-7b-hf \
    --image path/to/image.jpg \
    --question "How many objects are there in this image?"
```

Описание объекта по координатам bbox:

```bash
python vlm_inference/vlm_inference.py \
    --model_path ./vlm_model \
    --base_model llava-hf/llava-1.5-7b-hf \
    --image path/to/image.jpg \
    --bbox 100 200 300 400 \
    --label glass
```

**Параметры:**
- `--model_path`: путь к обученной модели (или LoRA весам)
- `--base_model`: базовая модель (если использовался LoRA)
- `--image`: путь к изображению
- `--question`: вопрос для модели
- `--bbox`: координаты bbox [x1, y1, x2, y2] (с --label)
- `--label`: метка объекта (используется с --bbox)

## Примеры использования VLM

### Пример 1: Описание объекта на траве

```python
from vlm_inference.vlm_inference import VLMInference

vlm = VLMInference(
    model_path="./vlm_model",
    base_model="llava-hf/llava-1.5-7b-hf"
)

# Описание glass объекта
answer = vlm.describe_object(
    "image.jpg",
    bbox=[150, 200, 300, 400],
    label="glass"
)
print(answer)  # "Here is a glass on the grass."
```

### Пример 2: Ответы на вопросы

```python
# Сколько объектов?
answer = vlm.answer_question(
    "image.jpg",
    "How many objects are there in this image?"
)

# Есть ли пластик?
answer = vlm.answer_question(
    "image.jpg",
    "Is there any plastic in this image?"
)
```

## Классы объектов

Модель детекции распознает следующие классы:
- `glass` - стекло
- `plastic` - пластик
- `metal` - металл
- `paper` - бумага
- `organic` - органические отходы
- `garb-garbage` - прочий мусор

## Ансамбль моделей детекции

Проект использует ансамбль из двух моделей:
1. **YOLOv8** - быстрая детекция объектов
2. **RT-DETR** - точная детекция с трансформером

Ансамбль объединяет результаты обеих моделей с учетом весов для каждого класса, что повышает точность детекции.

## Требования к ресурсам

### Для разметки:
- GPU: рекомендуется (ускоряет детекцию и генерацию описаний)
- RAM: минимум 8GB
- Диск: зависит от размера датасета

### Для обучения VLM:
- GPU: NVIDIA с минимум 14GB VRAM (для 7B модели)
- RAM: минимум 16GB
- Диск: ~50GB для модели и данных

### Для инференса:
- GPU: рекомендуется (но можно на CPU)
- RAM: минимум 8GB
- VRAM: зависит от размера модели

## Полный пайплайн

Используйте скрипт `run_full_pipeline.sh` для автоматического выполнения всех шагов:

1. Скачивание датасета Roboflow
2. Разметка всех сплитов (train/valid/test)
3. Конвертация в формат LLaVA
4. Обучение VLM модели

```bash
# Отредактируйте скрипт, указав правильные пути к моделям
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

## Работа с Git

### Обновление проекта в Git (после изменений):

```bash
# Быстрый вариант
git add . && git commit -m "Update VLM pipeline" && git push origin main

# Или по шагам
git add .
git commit -m "Your commit message"
git push origin main
```

### Обновление проекта из Git:

```bash
git pull origin main
```

Подробные инструкции: см. [git_commands.md](git_commands.md)

## Примечания

1. При использовании LoRA для обучения, при инференсе нужно указывать `--base_model`
2. Для уменьшения использования памяти можно использовать quantization (8-bit или 4-bit)
3. Рекомендуется использовать валидационный набор для мониторинга переобучения
4. Можно настроить параметры LoRA (r, alpha) для баланса между качеством и скоростью обучения

## Лицензия

[Укажите вашу лицензию]
