"""
ПОЛНАЯ VLM архитектура
- Ваш ансамбль (YOLO + RT-DETR) для детекции мусора
- Ваш классификатор сцен для определения окружения (YOLO или MobileNet)

Результат: "There is plastic on the grass"
"""

import sys
sys.path.append('.')
from PIL import Image
from pathlib import Path
import torch

from vlm_annotation.ensemble_detector import EnsembleDetector

# Классы сцен
SCENE_CLASSES = ['grass', 'marshy', 'rocky', 'sandy']


def load_scene_classifier(model_path: str):
    """
    Загрузка классификатора сцен (автоопределение типа: YOLO или MobileNet)
    """
    if model_path is None or not Path(model_path).exists():
        return None
    
    # Определяем тип модели по пути
    if 'yolo' in model_path.lower():
        from train_scene_yolo import SceneClassifierYOLO
        return SceneClassifierYOLO(model_path)
    else:
        from train_scene_classifier import SceneClassifierInference
        return SceneClassifierInference(model_path)


class CompleteVLM:
    """
    VLM где ВСЯ CV часть - это ваши модели:
    1. Ансамбль детекторов мусора
    2. Классификатор сцен
    """
    
    def __init__(
        self,
        yolo_path: str,
        detr_path: str,
        detr_processor_path: str,
        scene_classifier_path: str = None,
        conf_threshold: float = 0.5
    ):
        # Ваш ансамбль для мусора
        self.detector = EnsembleDetector(
            yolo_model_path=yolo_path,
            detr_model_path=detr_path,
            detr_processor_path=detr_processor_path,
            conf_threshold=conf_threshold
        )
        self.garbage_classes = ['glass', 'plastic', 'metal', 'paper', 'organic']
        print("✅ Ансамбль детекторов загружен!")
        
        # Ваш классификатор сцен (YOLO или MobileNet)
        self.scene_classifier = None
        if scene_classifier_path and Path(scene_classifier_path).exists():
            self.scene_classifier = load_scene_classifier(scene_classifier_path)
            classifier_type = "YOLO" if 'yolo' in scene_classifier_path.lower() else "MobileNet"
            print(f"✅ Классификатор сцен загружен ({classifier_type})!")
        else:
            print("⚠️ Классификатор сцен не найден. Описание сцены недоступно.")
            print("   Обучите его: python train_scene_yolo.py")
    
    def detect_garbage(self, image):
        """Детекция мусора вашим ансамблем"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        return self.detector.detect(image)
    
    def classify_scene(self, image):
        """Классификация сцены вашим классификатором"""
        if self.scene_classifier is None:
            return {'class': 'unknown', 'confidence': 0.0}
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        return self.scene_classifier.predict(image)
    
    def get_garbage_counts(self, detections):
        """Подсчёт объектов мусора"""
        counts = {cls: 0 for cls in self.garbage_classes}
        for det in detections:
            label = det['label']
            if label in counts:
                counts[label] += 1
        return counts
    
    def describe(self, image_path):
        """
        Полное описание: мусор + сцена
        Пример: "There is 2 plastic objects on the grass"
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Детекция мусора
        detections = self.detector.detect(image)
        counts = self.get_garbage_counts(detections)
        total = sum(counts.values())
        
        # Классификация сцены
        scene = self.classify_scene(image)
        scene_name = scene['class']
        scene_conf = scene['confidence']
        
        # Формируем описание мусора
        if total == 0:
            garbage_str = "No garbage"
        else:
            items = []
            for cls, count in counts.items():
                if count > 0:
                    items.append(f"{count} {cls}")
            garbage_str = "There is " + ", ".join(items)
        
        # Формируем полное описание
        # Порог 0.8 - если модель не уверена, не упоминаем сцену
        if scene_name != 'unknown' and scene_conf >= 0.8:
            preposition = self._get_preposition(scene_name)
            description = f"{garbage_str} {preposition} the {scene_name}."
        else:
            description = f"{garbage_str} detected."
        
        return description
    
    def _get_preposition(self, scene):
        """Правильный предлог для сцены"""
        if scene in ['marshy']:
            return 'in'
        elif scene in ['rocky']:
            return 'on'
        elif scene in ['sandy']:
            return 'on'
        else:  # grass и другие
            return 'on'
    
    def answer(self, image_path, question):
        """Ответ на вопрос о мусоре и сцене"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        detections = self.detector.detect(image)
        counts = self.get_garbage_counts(detections)
        total = sum(counts.values())
        q = question.lower()
        
        # Вопросы о сцене
        if any(word in q for word in ['where', 'scene', 'surface', 'ground', 'location']):
            scene = self.classify_scene(image)
            if scene['class'] != 'unknown' and scene['confidence'] >= 0.8:
                return f"The scene is: {scene['class']} ({scene['confidence']:.0%} confidence)."
            elif scene['class'] != 'unknown':
                return "Scene classification uncertain (confidence below 80%)."
            else:
                return "Scene classifier not available. Train it first!"
        
        # Вопросы о мусоре
        if "what" in q or "describe" in q:
            return self.describe(image)
        
        if "how many" in q:
            for cls in self.garbage_classes:
                if cls in q:
                    c = counts[cls]
                    return f"{c} {cls} object{'s' if c != 1 else ''} detected."
            return f"{total} garbage object{'s' if total != 1 else ''} detected in total."
        
        # Проверка наличия
        for cls in self.garbage_classes:
            if cls in q:
                c = counts[cls]
                if c > 0:
                    return f"Yes, {c} {cls} object{'s' if c != 1 else ''} detected."
                else:
                    return f"No {cls} detected."
        
        # Вопросы о типах сцен
        for scene_cls in SCENE_CLASSES:
            if scene_cls in q:
                scene = self.classify_scene(image)
                if scene['class'] == scene_cls and scene['confidence'] >= 0.8:
                    return f"Yes, the scene appears to be {scene_cls} ({scene['confidence']:.0%})."
                elif scene['confidence'] >= 0.8:
                    return f"No, the scene is classified as {scene['class']}, not {scene_cls}."
                else:
                    return "Scene classification uncertain (confidence below 80%)."
        
        # По умолчанию
        return self.describe(image)
    
    def get_full_analysis(self, image_path):
        """Полный анализ изображения"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        detections = self.detector.detect(image)
        scene = self.classify_scene(image)
        counts = self.get_garbage_counts(detections)
        
        return {
            'garbage': {
                'detections': detections,
                'counts': counts,
                'total': sum(counts.values())
            },
            'scene': scene,
            'description': self.describe(image)
        }


# === Для запуска ===
if __name__ == "__main__":
    import random
    
    print("\n" + "="*60)
    print("ПОЛНАЯ VLM АРХИТЕКТУРА")
    print("CV = Ваш ансамбль + Ваш классификатор сцен (YOLO)")
    print("="*60)
    
    # Приоритет: YOLO сцена > MobileNet сцена
    scene_path = None
    if Path("models/scene_classifier_yolo.pt").exists():
        scene_path = "models/scene_classifier_yolo.pt"
    elif Path("models/scene_classifier.pt").exists():
        scene_path = "models/scene_classifier.pt"
    
    vlm = CompleteVLM(
        yolo_path="models/yolo/yolov8x/best.pt",
        detr_path="models/rt-detr/rt-detr-101/m",
        detr_processor_path="models/rt-detr/rt-detr-101/p",
        scene_classifier_path=scene_path,
        conf_threshold=0.5
    )
    
    # Тест
    valid_dir = Path("data/roboflow_dataset/valid")
    test_dir = Path("data/roboflow_dataset/test")
    images = list(valid_dir.glob("*.jpg")) + list(test_dir.glob("*.jpg"))
    
    if images:
        img = random.choice(images)
        print(f"\n📷 Тест: {img.name}")
        
        analysis = vlm.get_full_analysis(str(img))
        
        print(f"\n🎯 Мусор: {analysis['garbage']['total']} объектов")
        for cls, count in analysis['garbage']['counts'].items():
            if count > 0:
                print(f"   - {cls}: {count}")
        
        print(f"\n🌍 Сцена: {analysis['scene']['class']} ({analysis['scene']['confidence']:.0%})")
        print(f"\n📝 Описание: {analysis['description']}")
        
        print("\n💬 Вопросы:")
        for q in ["Is there plastic?", "How many objects?", "Where is it?"]:
            print(f"   Q: {q}")
            print(f"   A: {vlm.answer(str(img), q)}")

