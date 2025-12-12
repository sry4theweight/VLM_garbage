# Загрузка проекта в Git (включая модели)

## Шаг 1: Установить и настроить Git LFS

```powershell
# Установить Git LFS (один раз)
git lfs install

# Настроить отслеживание больших файлов
git lfs track "*.pt"
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.onnx"

# Добавить .gitattributes
git add .gitattributes
```

## Шаг 2: Добавить все файлы и сделать push

```powershell
# Добавить все изменения
git add .

# Проверить что добавляется
git status

# Сделать commit
git commit -m "Add full project with models and training data"

# Push в репозиторий
git push origin main
```

## Шаг 3: В Google Colab

```python
# Клонировать репозиторий
!git lfs install
!git clone https://github.com/sry4theweight/VLM_garbage.git
%cd VLM_garbage

# Установить зависимости
!pip install -q transformers peft accelerate ultralytics supervision tqdm pillow roboflow

# Проверить GPU
!python check_gpu.py

# Скачать датасет Roboflow (если нужно для аннотации)
!python download_roboflow_dataset.py

# Запустить обучение (данные уже в репозитории)
!python train_pipeline.py --batch_size 2
```

## Примечания

- Git LFS нужен для файлов > 100MB
- Roboflow dataset скачивается отдельно (слишком большой)
- Модели YOLO и RT-DETR будут в репозитории через LFS

