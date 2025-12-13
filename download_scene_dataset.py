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


def prepare_roboflow_multiclass(dataset_dir: str, output_dir: str = "data/scene_dataset"):
    """
    Подготовка Roboflow multiclass датасета для обучения.
    
    Roboflow multiclass формат (CSV с one-hot encoding):
    dataset/
    ├── train/
    │   ├── _classes.csv  (filename, class1, class2, ...)
    │   └── *.jpg
    ├── valid/
    └── test/
    """
    import csv
    
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    print(f"\n📂 Анализ структуры датасета в {dataset_dir}...")
    
    # Ищем папки train/valid/test
    splits_map = {}
    for subdir in dataset_dir.iterdir():
        if subdir.is_dir():
            name = subdir.name.lower()
            if 'train' in name:
                splits_map['train'] = subdir
            elif 'valid' in name or 'val' in name:
                splits_map['val'] = subdir
            elif 'test' in name:
                splits_map['test'] = subdir
    
    if not splits_map:
        # Может быть _classes.csv в корне
        if (dataset_dir / '_classes.csv').exists():
            splits_map['train'] = dataset_dir
    
    print(f"   Найдены splits: {list(splits_map.keys())}")
    
    # Читаем классы из первого CSV
    classes = []
    for split_name, split_dir in splits_map.items():
        csv_path = split_dir / '_classes.csv'
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                # Первая колонка - filename, остальные - классы
                classes = [col.strip() for col in header[1:]]
                print(f"   Классы из CSV: {classes}")
                break
    
    if not classes:
        print("❌ Не найден _classes.csv")
        return None
    
    # Создаём выходную структуру
    for split in ['train', 'val']:
        for cls in classes:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Обрабатываем каждый split
    total = 0
    stats = {split: {cls: 0 for cls in classes} for split in ['train', 'val']}
    
    for split_name, split_dir in splits_map.items():
        # Маппинг: test -> val для выходной структуры
        out_split = 'val' if split_name == 'test' else split_name
        if out_split not in ['train', 'val']:
            out_split = 'train'
        
        csv_path = split_dir / '_classes.csv'
        if not csv_path.exists():
            continue
        
        print(f"\n   Обработка {split_name}...")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Пропускаем заголовок
            
            for row in reader:
                if len(row) < 2:
                    continue
                
                filename = row[0].strip()
                values = [int(v.strip()) for v in row[1:]]
                
                # Находим класс (колонка с 1)
                class_idx = None
                for i, v in enumerate(values):
                    if v == 1:
                        class_idx = i
                        break
                
                if class_idx is None or class_idx >= len(classes):
                    continue
                
                class_name = classes[class_idx]
                
                # Ищем изображение
                src_img = split_dir / filename
                if not src_img.exists():
                    # Пробуем разные расширения
                    for ext in ['.jpg', '.jpeg', '.png']:
                        test_path = split_dir / (Path(filename).stem + ext)
                        if test_path.exists():
                            src_img = test_path
                            break
                
                if src_img.exists():
                    dst_dir = output_dir / out_split / class_name
                    shutil.copy(src_img, dst_dir / src_img.name)
                    stats[out_split][class_name] += 1
                    total += 1
    
    print(f"\n✅ Скопировано {total} изображений")
    
    # Статистика
    print("\n📊 Статистика:")
    for split in ['train', 'val']:
        print(f"\n  {split}:")
        for cls in classes:
            print(f"    {cls}: {stats[split][cls]}")
    
    return output_dir, classes


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
