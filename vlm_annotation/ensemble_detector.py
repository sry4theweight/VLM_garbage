"""
Ensemble detector для объединения результатов YOLO и RT-DETR моделей
"""

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
import supervision as sv


MODEL_WEIGHTS = {
    "YOLOv8x": {
        "plastic": 0.8967,
        "glass": 0.8847,
        "metal": 0.8705,
        "paper": 0.7299,
        "organic": 0.6151,
    },
    "RT-DETR-101": {
        "plastic": 0.8967,
        "glass": 0.8847,
        "metal": 0.8705,
        "paper": 0.7299,
        "organic": 0.6151,
    }
}

# Классы для игнорирования
IGNORED_CLASSES = {'garb-garbage'}

CONFIDENCE_DELTA_THRESHOLD = 0.5


class EnsembleDetector:
    """Ансамбль моделей детекции объектов"""
    
    def __init__(self, yolo_model_path: str, detr_model_path: str, detr_processor_path: str, 
                 device: str = None, conf_threshold: float = 0.3):
        """
        Args:
            yolo_model_path: путь к модели YOLO (.pt файл)
            detr_model_path: путь к модели RT-DETR (директория m/)
            detr_processor_path: путь к процессору RT-DETR (директория p/)
            device: устройство для вычислений ('cuda' или 'cpu')
            conf_threshold: порог уверенности для детекций
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.class_names = ['garb-garbage', 'glass', 'metal', 'organic', 'paper', 'plastic']
        
        # Загрузка YOLO
        print(f"Loading YOLO model from {yolo_model_path}...")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        
        # Загрузка RT-DETR
        print(f"Loading RT-DETR model from {detr_model_path}...")
        self.detr_processor = RTDetrImageProcessor.from_pretrained(
            detr_processor_path, local_files_only=True, use_fast=True
        )
        self.detr_model = RTDetrV2ForObjectDetection.from_pretrained(
            detr_model_path, local_files_only=True
        ).to(self.device)
        self.detr_model.eval()
        
    def calibrate_confidence(self, confidence: float, model_name: str, class_name: str) -> float:
        """Калибровка уверенности на основе весов модели"""
        weight = MODEL_WEIGHTS[model_name].get(class_name.lower(), 1.0)
        return confidence * np.sqrt(weight) if weight > 0 else confidence
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Вычисление IoU между двумя bbox"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def match_detections(self, yolo_detections: list, detr_detections: list, 
                        iou_threshold: float = 0.3) -> list:
        """Сопоставление детекций между YOLO и RT-DETR"""
        matched = []
        used_detr = set()
        
        for yolo_idx, yolo_det in enumerate(yolo_detections):
            best_match = None
            best_iou = 0
            
            for detr_idx, detr_det in enumerate(detr_detections):
                if detr_idx in used_detr:
                    continue
                
                iou = self.calculate_iou(yolo_det['box'], detr_det['box'])
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = detr_idx
            
            if best_match is not None:
                matched.append((yolo_idx, best_match))
                used_detr.add(best_match)
        
        return matched
    
    def ensemble_predictions(self, yolo_det: dict, detr_det: dict) -> tuple:
        """Объединение предсказаний двух моделей"""
        yolo_label = yolo_det['label']
        yolo_conf = yolo_det['confidence']
        detr_label = detr_det['label']
        detr_conf = detr_det['confidence']
        
        yolo_conf_calibrated = self.calibrate_confidence(yolo_conf, "YOLOv8x", yolo_label)
        detr_conf_calibrated = self.calibrate_confidence(detr_conf, "RT-DETR-101", detr_label)
        
        if abs(yolo_conf_calibrated - detr_conf_calibrated) > CONFIDENCE_DELTA_THRESHOLD:
            yolo_weight = MODEL_WEIGHTS["YOLOv8x"].get(yolo_label.lower(), 0)
            detr_weight = MODEL_WEIGHTS["RT-DETR-101"].get(detr_label.lower(), 0)
            
            if yolo_weight > detr_weight:
                return yolo_label, yolo_conf_calibrated, yolo_det['box']
            else:
                return detr_label, detr_conf_calibrated, detr_det['box']
        
        # Вычисление взвешенного среднего для всех классов
        combined_scores = {}
        for class_name in self.class_names:
            yolo_score = yolo_conf_calibrated if yolo_label == class_name else 0
            detr_score = detr_conf_calibrated if detr_label == class_name else 0
            yolo_w = MODEL_WEIGHTS["YOLOv8x"].get(class_name.lower(), 0)
            detr_w = MODEL_WEIGHTS["RT-DETR-101"].get(class_name.lower(), 0)
            
            if yolo_w + detr_w > 0:
                combined_scores[class_name] = (yolo_w * yolo_score + detr_w * detr_score) / (yolo_w + detr_w)
            else:
                combined_scores[class_name] = 0
        
        best_class_name, best_score = max(combined_scores.items(), key=lambda x: x[1])
        avg_box = yolo_det['box']  # Используем bbox от YOLO
        
        return best_class_name, best_score, avg_box
    
    def detect(self, image: Image.Image) -> list:
        """
        Детекция объектов на изображении
        
        Returns:
            Список словарей с детекциями:
            [
                {
                    'box': [x1, y1, x2, y2],
                    'label': 'glass',
                    'confidence': 0.95
                },
                ...
            ]
        """
        # Детекция YOLO
        yolo_results = self.yolo_model(image, verbose=False)
        yolo_detections_sv = sv.Detections.from_ultralytics(yolo_results[0])
        
        yolo_dets = [
            {
                'box': box.tolist(),
                'confidence': float(conf),
                'label': self.class_names[int(cls)]
            }
            for box, conf, cls in zip(
                yolo_detections_sv.xyxy,
                yolo_detections_sv.confidence,
                yolo_detections_sv.class_id
            )
            if conf >= self.conf_threshold
        ]
        
        # Детекция RT-DETR
        inputs = self.detr_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detr_model(**inputs)
        
        w, h = image.size
        results = self.detr_processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=self.conf_threshold
        )
        detr_detections_sv = sv.Detections.from_transformers(results[0])
        
        detr_dets = [
            {
                'box': box.tolist(),
                'confidence': float(conf),
                'label': self.class_names[int(cls)]
            }
            for box, conf, cls in zip(
                detr_detections_sv.xyxy,
                detr_detections_sv.confidence,
                detr_detections_sv.class_id
            )
            if conf >= self.conf_threshold
        ]
        
        # Объединение результатов
        matched = self.match_detections(yolo_dets, detr_dets)
        ensemble_detections = []
        matched_yolo = {m[0] for m in matched}
        matched_detr = {m[1] for m in matched}
        
        # Объединенные детекции
        for yolo_idx, detr_idx in matched:
            label, conf, box = self.ensemble_predictions(yolo_dets[yolo_idx], detr_dets[detr_idx])
            ensemble_detections.append({
                'box': box,
                'label': label,
                'confidence': float(conf)
            })
        
        # Несовпадающие детекции YOLO
        for idx, det in enumerate(yolo_dets):
            if idx not in matched_yolo:
                ensemble_detections.append({
                    'box': det['box'],
                    'label': det['label'],
                    'confidence': float(self.calibrate_confidence(det['confidence'], "YOLOv8x", det['label']))
                })
        
        # Несовпадающие детекции RT-DETR
        for idx, det in enumerate(detr_dets):
            if idx not in matched_detr:
                ensemble_detections.append({
                    'box': det['box'],
                    'label': det['label'],
                    'confidence': float(self.calibrate_confidence(det['confidence'], "RT-DETR-101", det['label']))
                })
        
        # Фильтрация игнорируемых классов (например, garb-garbage)
        filtered_detections = [
            det for det in ensemble_detections 
            if det['label'] not in IGNORED_CLASSES
        ]
        
        return filtered_detections

