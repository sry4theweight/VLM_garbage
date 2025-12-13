"""
VLM с ЧЕСТНОЙ архитектурой
ВСЯ CV часть = ваш ансамбль (YOLO + RT-DETR)
Никаких сторонних моделей для анализа изображений!
"""

import sys
sys.path.append('.')
from PIL import Image
from pathlib import Path
import random

# Только ваш ансамбль!
from vlm_annotation.ensemble_detector import EnsembleDetector


class HonestVLM:
    """VLM где ВСЯ CV часть - это ваш ансамбль"""
    
    def __init__(self, yolo_path, detr_path, detr_processor_path, conf_threshold=0.5):
        self.detector = EnsembleDetector(
            yolo_model_path=yolo_path,
            detr_model_path=detr_path,
            detr_processor_path=detr_processor_path,
            conf_threshold=conf_threshold
        )
        self.classes = ['glass', 'plastic', 'metal', 'paper', 'organic']
        print("✅ VLM загружен! CV часть = ваш ансамбль")
    
    def detect(self, image_path):
        """Получить детекции от вашего ансамбля"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        return self.detector.detect(image)
    
    def get_class_counts(self, detections):
        """Подсчёт объектов по классам"""
        counts = {cls: 0 for cls in self.classes}
        for det in detections:
            label = det['label']
            if label in counts:
                counts[label] += 1
        return counts
    
    def answer(self, image_path, question):
        """
        Ответ на вопрос ТОЛЬКО на основе детекций ансамбля.
        Никакого BLIP, никаких других моделей!
        """
        detections = self.detect(image_path)
        counts = self.get_class_counts(detections)
        total = sum(counts.values())
        q = question.lower()
        
        # Описание того, что найдено
        if "what" in q or "describe" in q:
            if total == 0:
                return "No garbage objects detected by the ensemble detector."
            found = [f"{c} {cls}" for cls, c in counts.items() if c > 0]
            return f"The ensemble detector found: {', '.join(found)}."
        
        # Количество
        if "how many" in q:
            for cls in self.classes:
                if cls in q:
                    c = counts[cls]
                    return f"{c} {cls} object{'s' if c != 1 else ''} detected."
            return f"{total} garbage object{'s' if total != 1 else ''} detected in total."
        
        # Есть ли объект
        for cls in self.classes:
            if cls in q:
                c = counts[cls]
                if c > 0:
                    return f"Yes, {c} {cls} object{'s' if c != 1 else ''} detected."
                else:
                    return f"No {cls} detected."
        
        # По умолчанию
        if total == 0:
            return "No garbage detected."
        found = [f"{c} {cls}" for cls, c in counts.items() if c > 0]
        return f"Detected: {', '.join(found)}."
    
    def describe(self, image_path):
        """
        Описание изображения ТОЛЬКО на основе детекций.
        
        ВАЖНО: Ансамбль обучен только на классы мусора.
        Он НЕ МОЖЕТ описать сцену (трава, дорога и т.д.)
        Для этого нужно дообучить ансамбль на классы окружения!
        """
        detections = self.detect(image_path)
        counts = self.get_class_counts(detections)
        total = sum(counts.values())
        
        if total == 0:
            return "No garbage objects detected by the ensemble."
        
        # Формируем описание
        items = []
        for cls, count in counts.items():
            if count > 0:
                if count == 1:
                    items.append(f"1 {cls} object")
                else:
                    items.append(f"{count} {cls} objects")
        
        description = f"The ensemble detector found {', '.join(items)}."
        
        # Добавляем информацию о позициях
        positions = []
        w, h = Image.open(image_path).size if isinstance(image_path, str) else image_path.size
        for det in detections[:5]:  # Первые 5
            box = det['box']
            cx = (box[0] + box[2]) / 2 / w
            cy = (box[1] + box[3]) / 2 / h
            
            # Определяем позицию
            if cy < 0.33:
                pos_y = "top"
            elif cy > 0.66:
                pos_y = "bottom"
            else:
                pos_y = "middle"
            
            if cx < 0.33:
                pos_x = "left"
            elif cx > 0.66:
                pos_x = "right"
            else:
                pos_x = "center"
            
            positions.append(f"{det['label']} at {pos_y}-{pos_x}")
        
        description += f" Positions: {'; '.join(positions)}."
        
        return description


# === Для запуска в Colab ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ЧЕСТНАЯ VLM АРХИТЕКТУРА")
    print("CV часть = ТОЛЬКО ваш ансамбль (YOLO + RT-DETR)")
    print("="*60)
    
    vlm = HonestVLM(
        yolo_path="models/yolo/yolov8x/best.pt",
        detr_path="models/rt-detr/rt-detr-101/m",
        detr_processor_path="models/rt-detr/rt-detr-101/p",
        conf_threshold=0.5
    )
    
    # Тест
    valid_dir = Path("data/roboflow_dataset/valid")
    test_dir = Path("data/roboflow_dataset/test")
    images = list(valid_dir.glob("*.jpg")) + list(test_dir.glob("*.jpg"))
    
    if images:
        img = random.choice(images)
        print(f"\n📷 Тест на: {img.name}")
        print(f"\n🔍 Описание:\n{vlm.describe(str(img))}")
        print(f"\n❓ Is there plastic?\n{vlm.answer(str(img), 'is there plastic?')}")
        print(f"\n❓ How many objects?\n{vlm.answer(str(img), 'how many objects?')}")

