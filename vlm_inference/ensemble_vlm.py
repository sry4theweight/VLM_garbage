"""
VLM с использованием ансамбля детекторов (YOLO + RT-DETR) и LLM

Архитектура:
1. Изображение → Ансамбль (YOLO + RT-DETR) → Детекции (bbox, label, confidence)
2. Детекции + Вопрос → LLM → Ответ

Это позволяет использовать ваши обученные детекторы как "глаза" модели.
"""

import torch
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append(str(Path(__file__).parent.parent))

from vlm_annotation.ensemble_detector import EnsembleDetector


class EnsembleVLM:
    """VLM на основе ансамбля детекторов + LLM"""
    
    def __init__(
        self,
        yolo_model_path: str = "models/yolo/yolov8x/best.pt",
        detr_model_path: str = "models/rt-detr/rt-detr-101/m",
        detr_processor_path: str = "models/rt-detr/rt-detr-101/p",
        llm_model: str = "microsoft/phi-2",  # Легкая LLM
        device: str = None,
        conf_threshold: float = 0.5
    ):
        """
        Args:
            yolo_model_path: путь к YOLO модели
            detr_model_path: путь к RT-DETR модели
            detr_processor_path: путь к RT-DETR процессору
            llm_model: LLM для генерации ответов
            device: устройство (cuda/cpu)
            conf_threshold: порог уверенности для детекций
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка ансамбля детекторов
        print("Loading ensemble detector...")
        self.detector = EnsembleDetector(
            yolo_model_path=yolo_model_path,
            detr_model_path=detr_model_path,
            detr_processor_path=detr_processor_path,
            device=self.device,
            conf_threshold=conf_threshold
        )
        
        # Загрузка LLM
        print(f"Loading LLM: {llm_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.llm.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("EnsembleVLM ready!")
    
    def detect_objects(self, image: Image.Image) -> list:
        """Детекция объектов на изображении"""
        return self.detector.detect(image)
    
    def format_detections_as_context(self, detections: list) -> str:
        """Форматирование детекций как текстовый контекст для LLM"""
        if not detections:
            return "No objects detected in the image."
        
        # Подсчёт по классам
        class_counts = {}
        for det in detections:
            label = det['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Формирование описания
        context_parts = []
        context_parts.append(f"Detected {len(detections)} objects in the image:")
        
        for label, count in class_counts.items():
            context_parts.append(f"- {count} {label} object{'s' if count > 1 else ''}")
        
        # Детальная информация
        context_parts.append("\nDetailed detections:")
        for i, det in enumerate(detections):
            bbox = det['box']
            context_parts.append(
                f"{i+1}. {det['label']} (confidence: {det['confidence']:.0%}) "
                f"at position [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
            )
        
        return "\n".join(context_parts)
    
    def answer_question(self, image_path: str, question: str) -> str:
        """
        Ответ на вопрос по изображению
        
        Args:
            image_path: путь к изображению
            question: вопрос пользователя
        
        Returns:
            Ответ модели
        """
        # Загрузка изображения
        image = Image.open(image_path).convert('RGB')
        
        # Детекция объектов
        detections = self.detect_objects(image)
        
        # Формирование контекста
        context = self.format_detections_as_context(detections)
        
        # Формирование промпта для LLM
        prompt = f"""You are a helpful assistant that answers questions about garbage/waste objects in images.

Image Analysis:
{context}

Question: {question}

Answer: """
        
        # Генерация ответа
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Декодирование ответа
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлечение только ответа (после "Answer: ")
        if "Answer: " in full_response:
            answer = full_response.split("Answer: ")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()
        
        # Останавливаемся на первом переводе строки или вопросе
        answer = answer.split('\n')[0].strip()
        if '?' in answer:
            answer = answer.split('?')[0] + '.' if answer.split('?')[0] else answer
        
        return answer
    
    def get_detections_info(self, image_path: str) -> dict:
        """Получить информацию о детекциях"""
        image = Image.open(image_path).convert('RGB')
        detections = self.detect_objects(image)
        
        class_counts = {}
        for det in detections:
            label = det['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return {
            "total_objects": len(detections),
            "class_counts": class_counts,
            "detections": detections
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM с ансамблем детекторов")
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению")
    parser.add_argument("--question", type=str, default="What objects are in this image?")
    parser.add_argument("--yolo_model", type=str, default="models/yolo/yolov8x/best.pt")
    parser.add_argument("--detr_model", type=str, default="models/rt-detr/rt-detr-101/m")
    parser.add_argument("--detr_processor", type=str, default="models/rt-detr/rt-detr-101/p")
    parser.add_argument("--llm_model", type=str, default="microsoft/phi-2")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Создание модели
    vlm = EnsembleVLM(
        yolo_model_path=args.yolo_model,
        detr_model_path=args.detr_model,
        detr_processor_path=args.detr_processor,
        llm_model=args.llm_model,
        conf_threshold=args.conf_threshold
    )
    
    # Получение ответа
    print(f"\nImage: {args.image}")
    print(f"Question: {args.question}")
    
    # Сначала показываем детекции
    info = vlm.get_detections_info(args.image)
    print(f"\n📊 Detections: {info['total_objects']} objects")
    for label, count in info['class_counts'].items():
        print(f"   - {label}: {count}")
    
    # Затем ответ
    answer = vlm.answer_question(args.image, args.question)
    print(f"\n💬 Answer: {answer}")


if __name__ == "__main__":
    main()

