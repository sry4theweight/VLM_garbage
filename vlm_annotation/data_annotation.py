"""
Скрипт для разметки изображений с использованием ансамбля моделей детекции и LLM
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

from ensemble_detector import EnsembleDetector


class VLMDataAnnotator:
    """Аннотатор данных для обучения VLM"""
    
    def __init__(
        self,
        ensemble_detector: EnsembleDetector,
        llm_model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = None
    ):
        """
        Args:
            ensemble_detector: детектор объектов
            llm_model_name: название LLM модели из HuggingFace
            device: устройство для вычислений
        """
        self.detector = ensemble_detector
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading LLM model: {llm_model_name}...")
        self.llm_processor = BlipProcessor.from_pretrained(llm_model_name)
        self.llm_model = BlipForConditionalGeneration.from_pretrained(llm_model_name).to(self.device)
        self.llm_model.eval()
    
    def crop_bbox(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """Обрезка области bbox из изображения"""
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))
    
    def generate_description(self, image: Image.Image, label: str, bbox: List[float]) -> str:
        """
        Генерация текстового описания для объекта с учетом контекста
        
        Args:
            image: исходное изображение
            label: метка объекта (например, 'glass')
            bbox: координаты bbox [x1, y1, x2, y2]
        
        Returns:
            Текстовое описание (например, "here is a glass on the grass")
        """
        # Генерируем описание с контекстом всего изображения
        # Используем промпт который просит описать расположение объекта
        full_image_prompt = f"a {label} in this image. Describe where it is located and what surface or environment it is on."
        full_inputs = self.llm_processor(image, full_image_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            full_out = self.llm_model.generate(
                **full_inputs, 
                max_length=150, 
                num_beams=5,
                do_sample=True,
                temperature=0.7
            )
        
        full_description = self.llm_processor.decode(full_out[0], skip_special_tokens=True)
        
        # Форматируем описание в нужный формат: "here is a {label} on the {surface}"
        # Если описание не содержит информацию о поверхности, добавляем базовое описание
        full_description_lower = full_description.lower()
        
        # Проверяем, содержит ли описание информацию о расположении
        location_keywords = ['on', 'in', 'at', 'near', 'next to', 'beside', 'above', 'over', 'under']
        has_location = any(keyword in full_description_lower for keyword in location_keywords)
        
        if has_location:
            # Используем сгенерированное описание
            return f"Here is a {label}. {full_description_lower}"
        else:
            # Если нет информации о расположении, генерируем более детальное описание
            detail_prompt = f"a {label} bottle. What is it sitting on or surrounded by?"
            detail_inputs = self.llm_processor(image, detail_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                detail_out = self.llm_model.generate(
                    **detail_inputs,
                    max_length=100,
                    num_beams=3
                )
            detail_desc = self.llm_processor.decode(detail_out[0], skip_special_tokens=True).lower()
            
            # Формируем финальное описание
            if detail_desc and len(detail_desc) > 10:
                return f"Here is a {label}. {detail_desc}"
            else:
                return f"Here is a {label}."
    
    def generate_qa_pairs(self, image: Image.Image, detections: List[Dict]) -> List[Dict[str, str]]:
        """
        Генерация вопросов и ответов для VQA
        
        Args:
            image: изображение
            detections: список детекций
        
        Returns:
            Список словарей с вопросами и ответами
        """
        qa_pairs = []
        
        # Подсчет объектов по классам
        class_counts = {}
        for det in detections:
            label = det['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_objects = len(detections)
        
        # Генерация вопросов о количестве объектов
        if total_objects > 0:
            qa_pairs.append({
                "question": "How many objects are there in this image?",
                "answer": f"There are {total_objects} objects in this image."
            })
            
            for class_name, count in class_counts.items():
                qa_pairs.append({
                    "question": f"How many {class_name} objects are there?",
                    "answer": f"There {'is' if count == 1 else 'are'} {count} {class_name} object{'s' if count != 1 else ''}."
                })
                
                qa_pairs.append({
                    "question": f"Is there any {class_name} in this image?",
                    "answer": f"Yes, there {'is' if count == 1 else 'are'} {count} {class_name} object{'s' if count != 1 else ''}."
                })
        
        # Генерация вопросов о наличии конкретных классов
        all_classes = ['glass', 'plastic', 'metal', 'paper', 'organic', 'garb-garbage']
        for class_name in all_classes:
            if class_name not in class_counts:
                qa_pairs.append({
                    "question": f"Is there any {class_name} in this image?",
                    "answer": f"No, there is no {class_name} in this image."
                })
        
        return qa_pairs
    
    def annotate_image(self, image_path: str) -> Dict[str, Any]:
        """
        Разметка одного изображения
        
        Returns:
            Словарь с разметкой в формате для VLM:
            {
                "image": "path/to/image.jpg",
                "detections": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "label": "glass",
                        "confidence": 0.95,
                        "description": "here is a glass on the grass"
                    },
                    ...
                ],
                "qa_pairs": [
                    {"question": "...", "answer": "..."},
                    ...
                ]
            }
        """
        image = Image.open(image_path).convert("RGB")
        
        # Детекция объектов
        detections = self.detector.detect(image)
        
        # Генерация описаний для каждого объекта
        annotated_detections = []
        for det in detections:
            description = self.generate_description(image, det['label'], det['box'])
            annotated_detections.append({
                "bbox": det['box'],
                "label": det['label'],
                "confidence": det['confidence'],
                "description": description
            })
        
        # Генерация Q&A пар
        qa_pairs = self.generate_qa_pairs(image, detections)
        
        return {
            "image": image_path,
            "detections": annotated_detections,
            "qa_pairs": qa_pairs
        }
    
    def annotate_directory(self, images_dir: str, output_file: str, 
                          image_extensions: List[str] = None):
        """
        Разметка всех изображений в директории
        
        Args:
            images_dir: путь к директории с изображениями
            output_file: путь к выходному JSON файлу
            image_extensions: список расширений изображений (по умолчанию: ['.jpg', '.jpeg', '.png'])
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        images_dir = Path(images_dir)
        image_files = [
            f for f in images_dir.iterdir()
            if f.suffix in image_extensions
        ]
        
        print(f"Found {len(image_files)} images to annotate")
        
        annotations = []
        for image_file in tqdm(image_files, desc="Annotating images"):
            try:
                annotation = self.annotate_image(str(image_file))
                annotations.append(annotation)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        # Сохранение в JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"Annotations saved to {output_file}")
        print(f"Total annotated images: {len(annotations)}")


def main():
    parser = argparse.ArgumentParser(description="Разметка изображений для обучения VLM")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Путь к директории с изображениями"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Путь к выходному JSON файлу"
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
        "--device",
        type=str,
        default=None,
        help="Устройство для вычислений (cuda/cpu). По умолчанию автоматически"
    )
    
    args = parser.parse_args()
    
    # Создание детектора
    detector = EnsembleDetector(
        yolo_model_path=args.yolo_model,
        detr_model_path=args.detr_model,
        detr_processor_path=args.detr_processor,
        device=args.device,
        conf_threshold=args.conf_threshold
    )
    
    # Создание аннотатора
    annotator = VLMDataAnnotator(
        ensemble_detector=detector,
        llm_model_name=args.llm_model,
        device=args.device
    )
    
    # Разметка
    annotator.annotate_directory(args.images_dir, args.output_file)


if __name__ == "__main__":
    main()

