"""
Скрипт для разметки датасета Roboflow для обучения VLM
"""

import argparse
import json
import random
from pathlib import Path
from vlm_annotation.data_annotation import VLMDataAnnotator
from vlm_annotation.ensemble_detector import EnsembleDetector


def collect_all_images(dataset_path: Path, image_extensions: list = None, 
                       skip_augmented: bool = True) -> list:
    """
    Собрать все изображения из датасета
    
    Args:
        dataset_path: путь к датасету
        image_extensions: расширения файлов
        skip_augmented: пропустить папку train (содержит аугментированные изображения)
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    all_images = []
    
    # Если skip_augmented=True, берём только valid и test (без аугментаций)
    # train в Roboflow обычно содержит аугментированные копии
    if skip_augmented:
        subdirs = ['valid', 'test']
        print("📌 Используются только valid и test (без аугментаций)")
    else:
        subdirs = ['train', 'valid', 'test']
    
    for subdir in subdirs:
        subdir_path = dataset_path / subdir
        if subdir_path.exists():
            count = 0
            for f in subdir_path.iterdir():
                if f.suffix in image_extensions:
                    all_images.append(f)
                    count += 1
            print(f"   {subdir}: {count} изображений")
    
    # Также проверяем изображения в корне
    for f in dataset_path.iterdir():
        if f.is_file() and f.suffix in image_extensions:
            all_images.append(f)
    
    return all_images


def annotate_roboflow_dataset(
    roboflow_dataset_dir: str,
    output_dir: str,
    yolo_model: str,
    detr_model: str,
    detr_processor: str,
    llm_model: str = "Salesforce/blip-image-captioning-large",
    conf_threshold: float = 0.6,
    max_images: int = None,
    seed: int = 42,
    skip_augmented: bool = True
):
    """
    Разметка датасета Roboflow (все изображения в один файл)
    
    Args:
        roboflow_dataset_dir: путь к директории датасета Roboflow
        output_dir: директория для сохранения разметки
        yolo_model: путь к модели YOLO
        detr_model: путь к модели RT-DETR
        detr_processor: путь к процессору RT-DETR
        llm_model: модель LLM для генерации описаний
        conf_threshold: порог уверенности (по умолчанию 0.6)
        max_images: максимальное количество изображений (None = все)
        seed: seed для воспроизводимости случайного выбора
        skip_augmented: пропустить train (аугментированные изображения)
    """
    # Устанавливаем seed для воспроизводимости
    random.seed(seed)
    
    dataset_path = Path(roboflow_dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Собираем изображения (по умолчанию без аугментированных из train)
    print("Collecting images from dataset...")
    all_images = collect_all_images(dataset_path, skip_augmented=skip_augmented)
    print(f"Found {len(all_images)} images total")
    
    # Случайная выборка если нужно
    if max_images is not None and max_images < len(all_images):
        all_images = random.sample(all_images, max_images)
        print(f"Randomly selected {max_images} images for annotation")
    
    # Создание детектора
    print("\nInitializing ensemble detector...")
    detector = EnsembleDetector(
        yolo_model_path=yolo_model,
        detr_model_path=detr_model,
        detr_processor_path=detr_processor,
        conf_threshold=conf_threshold
    )
    
    # Создание аннотатора
    print("Initializing VLM data annotator...")
    annotator = VLMDataAnnotator(
        ensemble_detector=detector,
        llm_model_name=llm_model
    )
    
    print(f"\n{'='*60}")
    print(f"Processing {len(all_images)} images...")
    print(f"{'='*60}")
    
    # Аннотирование
    output_file = output_path / "annotations.json"
    annotator.annotate_images(all_images, str(output_file))
    
    # Сохранение информации о разметке
    info = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_path),
        "output_file": str(output_file),
        "total_images": len(all_images),
        "model_config": {
            "yolo_model": yolo_model,
            "detr_model": detr_model,
            "detr_processor": detr_processor,
            "llm_model": llm_model,
            "conf_threshold": conf_threshold,
            "max_images": max_images
        }
    }
    
    info_file = output_path / "annotation_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Annotation completed!")
    print(f"Annotations saved to: {output_file}")
    print(f"Info saved to: {info_file}")
    print(f"{'='*60}")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Разметка датасета Roboflow для обучения VLM")
    parser.add_argument(
        "--roboflow_dataset_dir",
        type=str,
        default="data/roboflow_dataset",
        help="Путь к директории датасета Roboflow"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/vlm_annotations",
        help="Директория для сохранения разметки"
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="models/yolo/yolov8x/best.pt",
        help="Путь к модели YOLO"
    )
    parser.add_argument(
        "--detr_model",
        type=str,
        default="models/rt-detr/rt-detr-101/m",
        help="Путь к модели RT-DETR (директория m/)"
    )
    parser.add_argument(
        "--detr_processor",
        type=str,
        default="models/rt-detr/rt-detr-101/p",
        help="Путь к процессору RT-DETR (директория p/)"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        help="Название LLM модели из HuggingFace"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.6,
        help="Порог уверенности для детекций (по умолчанию 0.6)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=200,
        help="Максимальное количество изображений (по умолчанию: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для воспроизводимости случайного выбора изображений"
    )
    parser.add_argument(
        "--include_train",
        action="store_true",
        help="Включить папку train (содержит аугментированные изображения). По умолчанию используются только valid/test."
    )
    
    args = parser.parse_args()
    
    annotate_roboflow_dataset(
        roboflow_dataset_dir=args.roboflow_dataset_dir,
        output_dir=args.output_dir,
        yolo_model=args.yolo_model,
        detr_model=args.detr_model,
        detr_processor=args.detr_processor,
        llm_model=args.llm_model,
        skip_augmented=not args.include_train,
        conf_threshold=args.conf_threshold,
        max_images=args.max_images,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

