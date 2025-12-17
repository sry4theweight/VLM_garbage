"""
Обучение YOLOv8 для классификации сцен (terrain)

Классы: grass, marshy, rocky, sandy

Использование:
    python train_scene_yolo.py --epochs 50
"""

import os
import shutil
from pathlib import Path
import argparse
from ultralytics import YOLO
import yaml


# Классы сцен
SCENE_CLASSES = ['grass', 'marshy', 'rocky', 'sandy']


def prepare_dataset_for_yolo(
    source_dir: str = "data/scene_dataset",
    output_dir: str = "data/scene_yolo_dataset"
):
    """
    Подготовка датасета в формате YOLO classification
    
    YOLO classification ожидает структуру:
    dataset/
        train/
            class1/
                img1.jpg
            class2/
                img2.jpg
        val/
            class1/
            class2/
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Очищаем output если существует
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Создаём структуру
    for split in ['train', 'val']:
        for cls in SCENE_CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Копируем файлы
    total_copied = 0
    
    for split in ['train', 'val']:
        source_split = source_dir / split
        if not source_split.exists():
            print(f"⚠️ Папка {source_split} не найдена")
            continue
            
        for cls in SCENE_CLASSES:
            source_cls = source_split / cls
            if not source_cls.exists():
                continue
                
            dest_cls = output_dir / split / cls
            
            for img_path in list(source_cls.glob("*.jpg")) + list(source_cls.glob("*.png")):
                shutil.copy(img_path, dest_cls / img_path.name)
                total_copied += 1
    
    print(f"✅ Скопировано {total_copied} изображений в {output_dir}")
    
    # Статистика
    print("\n📊 Статистика датасета:")
    for split in ['train', 'val']:
        print(f"\n{split}:")
        for cls in SCENE_CLASSES:
            count = len(list((output_dir / split / cls).glob("*")))
            print(f"  {cls}: {count}")
    
    return str(output_dir)


def check_gpu():
    """Проверка и настройка GPU"""
    import torch
    
    print("\n" + "=" * 60)
    print("ПРОВЕРКА GPU")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA доступна! GPU: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3
            print(f"   GPU {i}: {props.name}")
            print(f"   • Память: {total_mem:.1f} GB")
            print(f"   • Compute Capability: {props.major}.{props.minor}")
        
        # Очищаем кэш
        torch.cuda.empty_cache()
        
        return 0  # device index
    else:
        print("❌ CUDA недоступна! Обучение на CPU будет медленным.")
        return 'cpu'


def train_scene_classifier_yolo(
    data_dir: str = "data/scene_yolo_dataset",
    model_name: str = "yolov8x-cls",  # EXTRA-LARGE classification model
    epochs: int = 50,
    imgsz: int = 640,  # Увеличенный размер для лучшего качества
    batch: int = 64,   # Большой batch для быстрого обучения
    output_dir: str = "models/scene_yolo"
):
    """
    Обучение YOLOv8 классификатора сцен
    
    Args:
        data_dir: путь к датасету в формате YOLO classification
        model_name: название модели (yolov8n-cls, yolov8s-cls, yolov8m-cls, yolov8l-cls, yolov8x-cls)
        epochs: количество эпох
        imgsz: размер изображения
        batch: размер батча
        output_dir: директория для сохранения модели
    """
    import torch
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ YOLO КЛАССИФИКАТОРА СЦЕН")
    print("=" * 60)
    
    # Проверяем GPU
    device = check_gpu()
    
    # Проверяем датасет
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"\n❌ Датасет не найден: {data_dir}")
        print("   Сначала скачайте датасет:")
        print("   python download_scene_dataset.py --download")
        return None
    
    # Проверяем наличие изображений
    train_dir = data_dir / "train"
    has_images = False
    if train_dir.exists():
        for cls_dir in train_dir.iterdir():
            if cls_dir.is_dir() and list(cls_dir.glob("*"))[:1]:
                has_images = True
                break
    
    if not has_images:
        print(f"\n❌ Нет изображений в {train_dir}")
        print("   Сначала скачайте датасет:")
        print("   python download_scene_dataset.py --download")
        return None
    
    # Загружаем модель
    print(f"\n📦 Загрузка модели: {model_name}")
    model = YOLO(f"{model_name}.pt")
    
    # Обучение с явным указанием GPU
    print(f"\n🚀 Запуск обучения...")
    print(f"   Датасет: {data_dir}")
    print(f"   Модель: {model_name} (EXTRA-LARGE)")
    print(f"   Эпох: {epochs}")
    print(f"   Размер изображения: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Device: {'GPU' if device == 0 else device}")
    
    # Параметры для максимальной загрузки GPU
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name="scene_classifier",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        device=device,           # Явно указываем GPU
        workers=8,               # Больше workers для загрузки данных
        amp=True,                # Mixed precision для ускорения
        cache=True,              # Кэшировать изображения в RAM/GPU
        optimizer='AdamW',       # Современный оптимизатор
        lr0=0.001,               # Learning rate
        lrf=0.01,                # Final LR ratio
        warmup_epochs=3,         # Warmup
        cos_lr=True,             # Cosine LR scheduler
        label_smoothing=0.1,     # Регуляризация
    )
    
    # Копируем лучшую модель
    best_model_path = Path(output_dir) / "scene_classifier" / "weights" / "best.pt"
    final_model_path = Path("models") / "scene_classifier_yolo.pt"
    
    if best_model_path.exists():
        shutil.copy(best_model_path, final_model_path)
        print(f"\n✅ Модель сохранена: {final_model_path}")
    
    return results


class SceneClassifierYOLO:
    """Инференс YOLO классификатора сцен"""
    
    def __init__(self, model_path: str = "models/scene_classifier_yolo.pt"):
        from ultralytics import YOLO
        
        print(f"Loading YOLO scene classifier from {model_path}...")
        self.model = YOLO(model_path)
        self.classes = SCENE_CLASSES
        print(f"✅ Классы: {self.classes}")
    
    def predict(self, image):
        """
        Предсказание класса сцены
        
        Args:
            image: путь к изображению или PIL Image
            
        Returns:
            dict с 'class', 'confidence', 'all_probs'
        """
        results = self.model(image, verbose=False)
        
        probs = results[0].probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        
        # Получаем все вероятности
        all_probs = {}
        for i, cls in enumerate(self.classes):
            if i < len(probs.data):
                all_probs[cls] = probs.data[i].item()
            else:
                all_probs[cls] = 0.0
        
        return {
            'class': self.classes[top1_idx] if top1_idx < len(self.classes) else 'unknown',
            'confidence': top1_conf,
            'all_probs': all_probs
        }


def test_classifier(model_path: str, test_dir: str = "data/scene_yolo_dataset/val"):
    """Тестирование классификатора"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ КЛАССИФИКАТОРА")
    print("=" * 60)
    
    classifier = SceneClassifierYOLO(model_path)
    
    test_dir = Path(test_dir)
    correct = 0
    total = 0
    
    for cls in SCENE_CLASSES:
        cls_dir = test_dir / cls
        if not cls_dir.exists():
            continue
            
        for img_path in list(cls_dir.glob("*.jpg"))[:10]:  # Тест на 10 изображениях
            pred = classifier.predict(str(img_path))
            is_correct = pred['class'] == cls
            correct += int(is_correct)
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {img_path.name}: {pred['class']} ({pred['confidence']:.2%}) [GT: {cls}]")
    
    if total > 0:
        print(f"\n📊 Accuracy: {correct}/{total} = {correct/total:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO scene classifier")
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare dataset")
    parser.add_argument("--source_dir", type=str, default="data/scene_dataset")
    parser.add_argument("--data_dir", type=str, default="data/scene_yolo_dataset")
    parser.add_argument("--model", type=str, default="yolov8x-cls", 
                        choices=["yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls"],
                        help="Модель (x = самая большая и точная)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64, help="Batch size (больше = быстрее, но больше памяти)")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения (больше = точнее)")
    parser.add_argument("--test", action="store_true", help="Test existing model")
    parser.add_argument("--model_path", type=str, default="models/scene_classifier_yolo.pt")
    
    args = parser.parse_args()
    
    if args.test:
        test_classifier(args.model_path, f"{args.data_dir}/val")
    elif args.prepare_only:
        prepare_dataset_for_yolo(args.source_dir, args.data_dir)
    else:
        # Подготовка датасета
        prepare_dataset_for_yolo(args.source_dir, args.data_dir)
        
        # Обучение
        train_scene_classifier_yolo(
            data_dir=args.data_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz
        )

