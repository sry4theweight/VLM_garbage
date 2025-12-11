# Быстрый старт

## 1. Обновление из Git (если проект уже в Git)

```bash
git pull origin main
```

## 2. Установка зависимостей

```bash
pip install -r requirements_vlm.txt
```

## 3. Скачивание датасета Roboflow

Установите API ключ Roboflow (можно найти в настройках Roboflow):

```bash
export ROBOFLOW_API_KEY="your-api-key-here"
```

Или передайте его напрямую в команду:

```bash
python download_roboflow_dataset.py \
    --api_key "your-api-key-here" \
    --workspace "kuivashev" \
    --project "my-normal-dataset" \
    --version 1 \
    --output_dir "data/roboflow_dataset"
```

## 4. Разметка датасета

```bash
python annotate_roboflow_dataset.py \
    --roboflow_dataset_dir "data/roboflow_dataset/my-normal-dataset-1" \
    --output_dir "data/vlm_annotations" \
    --yolo_model "models/yolo/yolov8x/best.pt" \
    --detr_model "models/rt-detr/rt-detr-101/m" \
    --detr_processor "models/rt-detr/rt-detr-101/p"
```

## 5. Полный пайплайн (все шаги вместе)

Отредактируйте `run_full_pipeline.sh`, указав правильные пути к моделям, затем:

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

## 6. Обновление в Git после изменений

```bash
# Быстрый вариант (одной командой)
git add . && git commit -m "Update project" && git push origin main

# Или по шагам (см. git_commands.md)
git add .
git commit -m "Your commit message"
git push origin main
```

## Подробные инструкции

- `README.md` - полная документация
- `VLM_GUIDE.md` - подробное руководство по VLM
- `git_commands.md` - все команды Git

