"""
Скачивание датасета для классификации сцен

Использует Roboflow Terrain Classification dataset
Формат: multiclass (папки по классам)
"""

import os
import shutil
from pathlib import Path

# API ключ Roboflow
ROBOFLOW_API_KEY = "IYwUdRPmNzuSjy8A6cOr"


def download_terrain_dataset(output_dir: str = "data/scene_dataset"):
    """Скачивание Terrain Classification dataset с Roboflow"""
    
    print("📥 Скачивание Terrain Classification dataset...")
    
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("my-workplace-jkvgm").project("terrain-classification-1cg5i")
        version = project.version(1)
        dataset = version.download("multiclass", location=output_dir)
        
        print(f"✅ Скачано в: {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def prepare_roboflow_multiclass(dataset_dir: str, output_dir: str = "data/scene_dataset_prepared"):
    """
    Подготовка Roboflow multiclass датасета для обучения.
    
    Roboflow multiclass формат:
    dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   └── class2/
    │       └── image3.jpg
    ├── valid/
    └── test/
    
    Если формат другой, эта функция его исправит.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    print(f"\n📂 Анализ структуры датасета в {dataset_dir}...")
    
    # Проверяем структуру
    train_dir = None
    valid_dir = None
    
    # Ищем train/valid папки
    for subdir in dataset_dir.iterdir():
        if subdir.is_dir():
            name = subdir.name.lower()
            if 'train' in name:
                train_dir = subdir
            elif 'valid' in name or 'val' in name:
                valid_dir = subdir
            elif 'test' in name:
                if valid_dir is None:
                    valid_dir = subdir
    
    if train_dir is None:
        # Может быть папки классов прямо в корне
        class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if class_dirs:
            print("   Найдены папки классов в корне, используем их как train")
            train_dir = dataset_dir
    
    print(f"   Train: {train_dir}")
    print(f"   Valid: {valid_dir}")
    
    # Определяем классы
    classes = set()
    if train_dir:
        for item in train_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Проверяем есть ли изображения внутри
                images = list(item.glob("*.jpg")) + list(item.glob("*.png")) + list(item.glob("*.jpeg"))
                if images:
                    classes.add(item.name)
    
    if not classes:
        print("❌ Не найдены классы. Проверьте структуру датасета.")
        print("\nСтруктура должна быть:")
        print("  dataset/train/class_name/images...")
        return None
    
    print(f"\n📋 Найденные классы: {sorted(classes)}")
    
    # Создаём выходную структуру
    for split in ['train', 'val']:
        for cls in classes:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Копируем файлы
    total = 0
    
    # Train
    if train_dir:
        for cls in classes:
            src = train_dir / cls
            dst = output_dir / "train" / cls
            if src.exists():
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    for img in src.glob(ext):
                        shutil.copy(img, dst / img.name)
                        total += 1
    
    # Valid
    if valid_dir:
        for cls in classes:
            src = valid_dir / cls
            dst = output_dir / "val" / cls
            if src.exists():
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    for img in src.glob(ext):
                        shutil.copy(img, dst / img.name)
                        total += 1
    
    print(f"\n✅ Скопировано {total} изображений")
    
    # Статистика
    print("\n📊 Статистика:")
    for split in ['train', 'val']:
        print(f"\n  {split}:")
        for cls in sorted(classes):
            count = len(list((output_dir / split / cls).glob("*")))
            print(f"    {cls}: {count}")
    
    return output_dir, list(classes)


def update_scene_classes(classes: list):
    """Обновляет список классов в train_scene_classifier.py"""
    
    classifier_file = Path("train_scene_classifier.py")
    if not classifier_file.exists():
        print("⚠️ train_scene_classifier.py не найден")
        return
    
    content = classifier_file.read_text(encoding='utf-8')
    
    # Находим и заменяем SCENE_CLASSES
    import re
    new_classes = repr(classes)
    content = re.sub(
        r"SCENE_CLASSES = \[.*?\]",
        f"SCENE_CLASSES = {new_classes}",
        content,
        flags=re.DOTALL
    )
    
    classifier_file.write_text(content, encoding='utf-8')
    print(f"\n✅ Обновлены классы в train_scene_classifier.py: {classes}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download Roboflow dataset")
    parser.add_argument("--prepare", type=str, help="Path to downloaded dataset to prepare")
    parser.add_argument("--output", type=str, default="data/scene_dataset")
    
    args = parser.parse_args()
    
    if args.download:
        # Скачиваем
        raw_path = download_terrain_dataset("data/scene_dataset_raw")
        
        if raw_path:
            # Подготавливаем
            result = prepare_roboflow_multiclass(raw_path, args.output)
            if result:
                output_dir, classes = result
                update_scene_classes(classes)
                print(f"\n🎉 Датасет готов в {output_dir}")
    elif args.prepare:
        result = prepare_roboflow_multiclass(args.prepare, args.output)
        if result:
            output_dir, classes = result
            update_scene_classes(classes)
    else:
        print("Использование:")
        print("  python download_scene_dataset.py --download")
        print("  python download_scene_dataset.py --prepare /path/to/dataset")
