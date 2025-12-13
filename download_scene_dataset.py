"""
Скачивание датасета для классификации сцен

Использует Intel Image Classification dataset:
- buildings → indoor/outdoor_other
- forest → grass
- glacier → outdoor_other  
- mountain → outdoor_other
- sea → water
- street → road

Этот датасет бесплатный и доступен через kagglehub
"""

import os
import shutil
from pathlib import Path

# Маппинг классов Intel → наши классы
CLASS_MAPPING = {
    'buildings': 'indoor',
    'forest': 'grass',
    'glacier': 'outdoor_other',
    'mountain': 'outdoor_other',
    'sea': 'water',
    'street': 'road'
}

def download_intel_dataset():
    """Скачивание датасета Intel Image Classification"""
    
    print("📥 Скачивание Intel Image Classification dataset...")
    print("   Этот датасет содержит ~25,000 изображений")
    
    try:
        import kagglehub
        
        # Скачиваем датасет
        path = kagglehub.dataset_download("puneet6060/intel-image-classification")
        print(f"✅ Скачано в: {path}")
        return path
        
    except ImportError:
        print("❌ kagglehub не установлен. Установите:")
        print("   pip install kagglehub")
        print("\nИЛИ скачайте вручную:")
        print("   https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\nПопробуйте скачать вручную:")
        print("   https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
        return None


def prepare_scene_dataset(intel_path: str, output_dir: str = "data/scene_dataset"):
    """
    Конвертирует Intel dataset в наш формат с нужными классами
    """
    intel_path = Path(intel_path)
    output_dir = Path(output_dir)
    
    # Создаём структуру
    for split in ['train', 'val']:
        for cls in set(CLASS_MAPPING.values()):
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Ищем папки с данными
    # Intel dataset structure: seg_train/seg_train/class_name/
    train_dirs = list(intel_path.glob("**/seg_train/seg_train"))
    test_dirs = list(intel_path.glob("**/seg_test/seg_test"))
    
    if not train_dirs:
        train_dirs = list(intel_path.glob("**/seg_train"))
    if not test_dirs:
        test_dirs = list(intel_path.glob("**/seg_test"))
    
    if not train_dirs:
        print(f"❌ Не найдены данные в {intel_path}")
        print("   Структура должна быть: seg_train/class_name/images...")
        return
    
    train_dir = train_dirs[0]
    test_dir = test_dirs[0] if test_dirs else None
    
    print(f"📂 Train: {train_dir}")
    print(f"📂 Test: {test_dir}")
    
    total_copied = 0
    
    # Копируем train
    for intel_class, our_class in CLASS_MAPPING.items():
        src_dir = train_dir / intel_class
        dst_dir = output_dir / "train" / our_class
        
        if src_dir.exists():
            for img in src_dir.glob("*.jpg"):
                shutil.copy(img, dst_dir / f"{intel_class}_{img.name}")
                total_copied += 1
            print(f"  {intel_class} → {our_class}: {len(list(src_dir.glob('*.jpg')))} images")
    
    # Копируем test → val
    if test_dir:
        for intel_class, our_class in CLASS_MAPPING.items():
            src_dir = test_dir / intel_class
            dst_dir = output_dir / "val" / our_class
            
            if src_dir.exists():
                for img in src_dir.glob("*.jpg"):
                    shutil.copy(img, dst_dir / f"{intel_class}_{img.name}")
                    total_copied += 1
    
    print(f"\n✅ Скопировано {total_copied} изображений в {output_dir}")
    
    # Статистика
    print("\n📊 Статистика датасета:")
    for split in ['train', 'val']:
        print(f"\n  {split}:")
        for cls in sorted(set(CLASS_MAPPING.values())):
            count = len(list((output_dir / split / cls).glob("*.jpg")))
            print(f"    {cls}: {count}")


def add_additional_classes(output_dir: str = "data/scene_dataset"):
    """
    Добавление дополнительных классов (sand, floor)
    Эти классы нужно добавить вручную или через Roboflow
    """
    output_dir = Path(output_dir)
    
    # Создаём папки для недостающих классов
    missing = ['sand', 'floor']
    
    for cls in missing:
        for split in ['train', 'val']:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    print("\n⚠️ Недостающие классы (нужно добавить вручную):")
    print("   - sand: изображения пляжей, песка")
    print("   - floor: изображения полов (плитка, паркет)")
    print("\nМожно найти на:")
    print("   - https://universe.roboflow.com/ (поиск: sand, floor)")
    print("   - https://unsplash.com/ (бесплатные фото)")
    print("   - Google Images")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download Intel dataset")
    parser.add_argument("--prepare", type=str, help="Path to downloaded Intel dataset")
    parser.add_argument("--output", type=str, default="data/scene_dataset")
    
    args = parser.parse_args()
    
    if args.download:
        path = download_intel_dataset()
        if path:
            prepare_scene_dataset(path, args.output)
            add_additional_classes(args.output)
    elif args.prepare:
        prepare_scene_dataset(args.prepare, args.output)
        add_additional_classes(args.output)
    else:
        print("Использование:")
        print("  python download_scene_dataset.py --download")
        print("  python download_scene_dataset.py --prepare /path/to/intel_dataset")

