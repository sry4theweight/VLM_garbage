"""
Скрипт для разметки всего датасета Roboflow (train/valid/test) для обучения VLM
"""

import argparse
import json
from pathlib import Path
from vlm_annotation.data_annotation import VLMDataAnnotator
from vlm_annotation.ensemble_detector import EnsembleDetector


def annotate_roboflow_dataset(
    roboflow_dataset_dir: str,
    output_dir: str,
    yolo_model: str,
    detr_model: str,
    detr_processor: str,
    llm_model: str = "Salesforce/blip-image-captioning-large",
    conf_threshold: float = 0.3,
    splits: list = None
):
    """
    Разметка датасета Roboflow для всех сплитов
    
    Args:
        roboflow_dataset_dir: путь к директории датасета Roboflow (например, data/roboflow_dataset/my-normal-dataset-1)
        output_dir: директория для сохранения разметки
        yolo_model: путь к модели YOLO
        detr_model: путь к модели RT-DETR
        detr_processor: путь к процессору RT-DETR
        llm_model: модель LLM для генерации описаний
        conf_threshold: порог уверенности
        splits: список сплитов для обработки (по умолчанию ['train', 'valid', 'test'])
    """
    if splits is None:
        splits = ['train', 'valid', 'test']
    
    dataset_path = Path(roboflow_dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Создание детектора
    print("Initializing ensemble detector...")
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
    
    all_annotations = {}
    
    # Обработка каждого сплита
    for split in splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist. Skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        # Разметка сплита
        output_file = output_path / f"{split}_annotations.json"
        annotator.annotate_directory(str(split_dir), str(output_file))
        
        all_annotations[split] = str(output_file)
    
    # Сохранение информации о разметке
    info = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_path),
        "splits": all_annotations,
        "model_config": {
            "yolo_model": yolo_model,
            "detr_model": detr_model,
            "detr_processor": detr_processor,
            "llm_model": llm_model,
            "conf_threshold": conf_threshold
        }
    }
    
    info_file = output_path / "annotation_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Annotation completed!")
    print(f"Annotation info saved to: {info_file}")
    print(f"{'='*60}")
    
    return all_annotations


def main():
    parser = argparse.ArgumentParser(description="Разметка датасета Roboflow для обучения VLM")
    parser.add_argument(
        "--roboflow_dataset_dir",
        type=str,
        required=True,
        help="Путь к директории датасета Roboflow (например, data/roboflow_dataset/my-normal-dataset-1)"
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
        default=0.3,
        help="Порог уверенности для детекций"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Сплиты для обработки (по умолчанию: train valid test)"
    )
    
    args = parser.parse_args()
    
    annotate_roboflow_dataset(
        roboflow_dataset_dir=args.roboflow_dataset_dir,
        output_dir=args.output_dir,
        yolo_model=args.yolo_model,
        detr_model=args.detr_model,
        detr_processor=args.detr_processor,
        llm_model=args.llm_model,
        conf_threshold=args.conf_threshold,
        splits=args.splits
    )


if __name__ == "__main__":
    main()

