"""
Модуль оценки метрик VLM системы

Оценивает:
1. Детектор мусора (ансамбль YOLO + RT-DETR) - mAP, Precision, Recall
2. Классификатор сцен (YOLO) - Accuracy, Precision, Recall, F1, Confusion Matrix
3. Полная VLM система - время инференса, статистика описаний

Запуск:
    python evaluate_vlm.py --max_images 100
"""

import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import torch

# Классы мусора
GARBAGE_CLASSES = ['glass', 'metal', 'organic', 'paper', 'plastic']

# Классы сцен
SCENE_CLASSES = ['grass', 'marshy', 'rocky', 'sandy']


def load_coco_annotations(annotation_file: str) -> dict:
    """Загрузка COCO аннотаций"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    
    annotations_by_image = defaultdict(list)
    for ann in coco['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']
        
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        category_name = categories.get(category_id, 'unknown')
        
        if category_name == 'garb-garbage':
            continue
            
        annotations_by_image[image_id].append({
            'bbox': [x1, y1, x2, y2],
            'label': category_name,
            'area': ann.get('area', w * h)
        })
    
    return {
        'images': images,
        'categories': categories,
        'annotations': annotations_by_image
    }


def calculate_iou(box1: list, box2: list) -> float:
    """Вычисление IoU между двумя bbox [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_predictions_to_gt(predictions: list, gt_annotations: list, iou_threshold: float = 0.5) -> dict:
    """Сопоставление предсказаний с ground truth"""
    results = {cls: {'tp': 0, 'fp': 0, 'fn': 0, 'confs': []} for cls in GARBAGE_CLASSES}
    
    gt_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)
    
    for gt in gt_annotations:
        gt_by_class[gt['label']].append(gt)
    
    for pred in predictions:
        pred_by_class[pred['label']].append(pred)
    
    for cls in GARBAGE_CLASSES:
        gt_boxes = gt_by_class[cls]
        pred_boxes = sorted(pred_by_class[cls], key=lambda x: x['confidence'], reverse=True)
        
        matched_gt = set()
        
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred['box'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                results[cls]['tp'] += 1
                results[cls]['confs'].append((pred['confidence'], True))
                matched_gt.add(best_gt_idx)
            else:
                results[cls]['fp'] += 1
                results[cls]['confs'].append((pred['confidence'], False))
        
        results[cls]['fn'] = len(gt_boxes) - len(matched_gt)
    
    return results


class DetectorEvaluator:
    """Оценка детектора мусора"""
    
    def __init__(self, detector):
        self.detector = detector
    
    def evaluate(self, annotation_file: str, images_dir: str, 
                 max_images: int = None, iou_threshold: float = 0.5) -> dict:
        """Оценка детектора на датасете с COCO аннотациями"""
        print(f"\n{'='*60}")
        print("📦 ОЦЕНКА ДЕТЕКТОРА МУСОРА (Ансамбль YOLO + RT-DETR)")
        print(f"{'='*60}")
        
        coco_data = load_coco_annotations(annotation_file)
        images = coco_data['images']
        annotations = coco_data['annotations']
        
        images_dir = Path(images_dir)
        
        image_ids = list(images.keys())
        if max_images:
            import random
            random.seed(42)
            image_ids = random.sample(image_ids, min(max_images, len(image_ids)))
        
        print(f"📊 Изображений для оценки: {len(image_ids)}")
        print(f"📊 IoU threshold: {iou_threshold}")
        
        all_results = {cls: {'tp': 0, 'fp': 0, 'fn': 0, 'confs': []} for cls in GARBAGE_CLASSES}
        inference_times = []
        
        for img_id in tqdm(image_ids, desc="Детекция"):
            img_info = images[img_id]
            img_path = images_dir / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            image = Image.open(img_path).convert('RGB')
            
            start_time = time.time()
            predictions = self.detector.detect(image)
            inference_times.append(time.time() - start_time)
            
            gt_anns = annotations.get(img_id, [])
            results = match_predictions_to_gt(predictions, gt_anns, iou_threshold)
            
            for cls in GARBAGE_CLASSES:
                all_results[cls]['tp'] += results[cls]['tp']
                all_results[cls]['fp'] += results[cls]['fp']
                all_results[cls]['fn'] += results[cls]['fn']
                all_results[cls]['confs'].extend(results[cls]['confs'])
        
        # Вычисляем метрики
        metrics = {}
        total_tp, total_fp, total_fn = 0, 0, 0
        
        print(f"\n{'Класс':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-" * 72)
        
        ap_scores = []
        for cls in GARBAGE_CLASSES:
            tp = all_results[cls]['tp']
            fp = all_results[cls]['fp']
            fn = all_results[cls]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # AP calculation
            confs = all_results[cls]['confs']
            if confs:
                confs_sorted = sorted(confs, key=lambda x: -x[0])
                cumsum_tp = 0
                cumsum_fp = 0
                precisions_at_recall = []
                total_positives = tp + fn
                
                for conf, is_tp in confs_sorted:
                    if is_tp:
                        cumsum_tp += 1
                    else:
                        cumsum_fp += 1
                    prec = cumsum_tp / (cumsum_tp + cumsum_fp)
                    rec = cumsum_tp / total_positives if total_positives > 0 else 0
                    precisions_at_recall.append((rec, prec))
                
                # Interpolated AP
                ap = 0
                for i in range(len(precisions_at_recall) - 1):
                    r1, p1 = precisions_at_recall[i]
                    r2, p2 = precisions_at_recall[i + 1]
                    ap += (r2 - r1) * max(p1, p2)
                if precisions_at_recall:
                    ap = max(ap, precisions_at_recall[-1][1] * precisions_at_recall[-1][0])
            else:
                ap = 0
            
            ap_scores.append(ap)
            
            metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap': ap,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            print(f"{cls:<12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {ap:>10.3f} {tp:>6} {fp:>6} {fn:>6}")
        
        # Общие метрики
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        mAP = np.mean(ap_scores)
        
        print("-" * 72)
        print(f"{'OVERALL':<12} {overall_precision:>10.3f} {overall_recall:>10.3f} {overall_f1:>10.3f} {mAP:>10.3f} {total_tp:>6} {total_fp:>6} {total_fn:>6}")
        
        print(f"\n📈 СВОДКА ДЕТЕКТОРА:")
        print(f"   • mAP@{iou_threshold}:        {mAP:.3f}")
        print(f"   • Precision:        {overall_precision:.3f}")
        print(f"   • Recall:           {overall_recall:.3f}")
        print(f"   • F1-score:         {overall_f1:.3f}")
        print(f"   • Время инференса:  {np.mean(inference_times)*1000:.1f} ± {np.std(inference_times)*1000:.1f} ms")
        
        return {
            'per_class': metrics,
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'mAP': mAP
            },
            'timing': {
                'avg_ms': np.mean(inference_times) * 1000,
                'std_ms': np.std(inference_times) * 1000
            },
            'counts': {
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'images': len(image_ids)
            }
        }


class SceneClassifierEvaluator:
    """Оценка YOLO классификатора сцен"""
    
    def __init__(self, classifier):
        self.classifier = classifier
    
    def evaluate(self, test_dir: str, max_images: int = None) -> dict:
        """Оценка классификатора сцен"""
        print(f"\n{'='*60}")
        print("🌍 ОЦЕНКА КЛАССИФИКАТОРА СЦЕН (YOLO)")
        print(f"{'='*60}")
        
        test_dir = Path(test_dir)
        
        samples = []
        for cls_idx, cls_name in enumerate(SCENE_CLASSES):
            cls_dir = test_dir / cls_name
            if cls_dir.exists():
                for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")):
                    samples.append((img_path, cls_idx, cls_name))
        
        if max_images and len(samples) > max_images:
            import random
            random.seed(42)
            samples = random.sample(samples, max_images)
        
        if not samples:
            print("⚠️ Нет изображений для оценки классификатора сцен")
            return None
        
        print(f"📊 Изображений для оценки: {len(samples)}")
        
        confusion = np.zeros((len(SCENE_CLASSES), len(SCENE_CLASSES)), dtype=int)
        inference_times = []
        confidences = []
        
        for img_path, true_idx, true_cls in tqdm(samples, desc="Классификация сцен"):
            image = Image.open(img_path).convert('RGB')
            
            start_time = time.time()
            pred = self.classifier.predict(image)
            inference_times.append(time.time() - start_time)
            
            pred_cls = pred['class']
            pred_conf = pred['confidence']
            confidences.append(pred_conf)
            
            if pred_cls in SCENE_CLASSES:
                pred_idx = SCENE_CLASSES.index(pred_cls)
                confusion[true_idx, pred_idx] += 1
        
        # Метрики
        accuracy = np.trace(confusion) / np.sum(confusion) if np.sum(confusion) > 0 else 0
        
        per_class_metrics = {}
        print(f"\n{'Класс':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 52)
        
        for idx, cls_name in enumerate(SCENE_CLASSES):
            tp = confusion[idx, idx]
            fp = np.sum(confusion[:, idx]) - tp
            fn = np.sum(confusion[idx, :]) - tp
            support = np.sum(confusion[idx, :])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[cls_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(support)
            }
            
            print(f"{cls_name:<12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")
        
        macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
        
        print("-" * 52)
        print(f"{'Accuracy':<12} {accuracy:>10.3f}")
        print(f"{'Macro F1':<12} {macro_f1:>10.3f}")
        
        # Confusion Matrix
        print(f"\n📊 CONFUSION MATRIX:")
        print(f"{'':>12}", end="")
        for cls in SCENE_CLASSES:
            print(f"{cls[:8]:>10}", end="")
        print()
        
        for i, cls in enumerate(SCENE_CLASSES):
            print(f"{cls:<12}", end="")
            for j in range(len(SCENE_CLASSES)):
                print(f"{confusion[i, j]:>10}", end="")
            print()
        
        print(f"\n📈 СВОДКА КЛАССИФИКАТОРА СЦЕН:")
        print(f"   • Accuracy:         {accuracy:.3f}")
        print(f"   • Macro F1:         {macro_f1:.3f}")
        print(f"   • Avg Confidence:   {np.mean(confidences):.3f}")
        print(f"   • Время инференса:  {np.mean(inference_times)*1000:.1f} ± {np.std(inference_times)*1000:.1f} ms")
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class': per_class_metrics,
            'confusion_matrix': confusion.tolist(),
            'classes': SCENE_CLASSES,
            'timing': {
                'avg_ms': np.mean(inference_times) * 1000,
                'std_ms': np.std(inference_times) * 1000
            },
            'confidence': {
                'mean': np.mean(confidences),
                'std': np.std(confidences)
            }
        }


class VLMEvaluator:
    """Оценка полной VLM системы"""
    
    def __init__(self, vlm):
        self.vlm = vlm
    
    def evaluate(self, images_dir: str, max_images: int = 50) -> dict:
        """Оценка полной VLM системы"""
        print(f"\n{'='*60}")
        print("🤖 ОЦЕНКА ПОЛНОЙ VLM СИСТЕМЫ")
        print(f"{'='*60}")
        
        images_dir = Path(images_dir)
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not images:
            print("⚠️ Нет изображений для оценки")
            return None
        
        import random
        random.seed(42)
        images = random.sample(images, min(max_images, len(images)))
        
        print(f"📊 Изображений для оценки: {len(images)}")
        
        results = []
        inference_times = []
        detection_stats = defaultdict(int)
        scene_stats = defaultdict(int)
        confident_scenes = 0
        
        for img_path in tqdm(images, desc="Анализ VLM"):
            start_time = time.time()
            analysis = self.vlm.get_full_analysis(str(img_path))
            inference_times.append(time.time() - start_time)
            
            for cls, count in analysis['garbage']['counts'].items():
                detection_stats[cls] += count
            
            scene = analysis['scene']['class']
            scene_conf = analysis['scene']['confidence']
            
            if scene_conf >= 0.8:
                scene_stats[scene] += 1
                confident_scenes += 1
            else:
                scene_stats['uncertain'] += 1
            
            results.append({
                'image': img_path.name,
                'description': analysis['description'],
                'garbage_total': analysis['garbage']['total'],
                'scene': scene,
                'scene_confidence': scene_conf
            })
        
        # Вывод статистики
        print(f"\n🗑️ СТАТИСТИКА ДЕТЕКЦИЙ:")
        total_detections = sum(detection_stats.values())
        print(f"   Всего объектов: {total_detections}")
        for cls in GARBAGE_CLASSES:
            count = detection_stats[cls]
            pct = count / total_detections * 100 if total_detections > 0 else 0
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"   {cls:<12}: {count:>5} ({pct:>5.1f}%) {bar}")
        
        print(f"\n🌍 СТАТИСТИКА СЦЕН:")
        total_scenes = len(images)
        for scene, count in sorted(scene_stats.items(), key=lambda x: -x[1]):
            pct = count / total_scenes * 100 if total_scenes > 0 else 0
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"   {scene:<12}: {count:>5} ({pct:>5.1f}%) {bar}")
        
        print(f"\n⏱️ ВРЕМЯ ИНФЕРЕНСА:")
        print(f"   Среднее:    {np.mean(inference_times)*1000:.1f} ms")
        print(f"   Медиана:    {np.median(inference_times)*1000:.1f} ms")
        print(f"   Мин:        {np.min(inference_times)*1000:.1f} ms")
        print(f"   Макс:       {np.max(inference_times)*1000:.1f} ms")
        print(f"   FPS:        {1/np.mean(inference_times):.1f}")
        
        print(f"\n📝 ПРИМЕРЫ ОПИСАНИЙ:")
        for r in results[:5]:
            print(f"   📷 {r['image']}")
            print(f"      \"{r['description']}\"")
            print()
        
        return {
            'detection_stats': dict(detection_stats),
            'scene_stats': dict(scene_stats),
            'confident_scenes_ratio': confident_scenes / len(images) if images else 0,
            'timing': {
                'avg_ms': np.mean(inference_times) * 1000,
                'median_ms': np.median(inference_times) * 1000,
                'min_ms': np.min(inference_times) * 1000,
                'max_ms': np.max(inference_times) * 1000,
                'fps': 1 / np.mean(inference_times)
            },
            'examples': results[:10]
        }


def evaluate_full_system(
    yolo_path: str = "models/yolo/yolov8x/best.pt",
    detr_path: str = "models/rt-detr/rt-detr-101/m",
    detr_processor_path: str = "models/rt-detr/rt-detr-101/p",
    scene_classifier_path: str = None,
    test_annotation_file: str = "data/roboflow_dataset/test/_annotations.coco.json",
    test_images_dir: str = "data/roboflow_dataset/test",
    scene_test_dir: str = "data/scene_yolo_dataset/val",
    max_images: int = 100,
    conf_threshold: float = 0.5,
    output_file: str = "evaluation_results.json"
):
    """Полная оценка системы"""
    import sys
    sys.path.append('.')
    
    print("\n" + "=" * 60)
    print("🔬 ПОЛНАЯ ОЦЕНКА VLM СИСТЕМЫ")
    print("=" * 60)
    print(f"\nКонфигурация:")
    print(f"   • YOLO детектор: {yolo_path}")
    print(f"   • RT-DETR: {detr_path}")
    print(f"   • Scene classifier: {scene_classifier_path}")
    print(f"   • Max images: {max_images}")
    print(f"   • Confidence threshold: {conf_threshold}")
    
    results = {}
    
    # Определяем путь к классификатору сцен
    if scene_classifier_path is None:
        if Path("models/scene_classifier_yolo.pt").exists():
            scene_classifier_path = "models/scene_classifier_yolo.pt"
        elif Path("models/scene_classifier.pt").exists():
            scene_classifier_path = "models/scene_classifier.pt"
    
    # 1. Оценка детектора
    print("\n\n[1/3] ДЕТЕКТОР МУСОРА")
    print("-" * 60)
    
    if Path(test_annotation_file).exists():
        from vlm_annotation.ensemble_detector import EnsembleDetector
        
        detector = EnsembleDetector(
            yolo_model_path=yolo_path,
            detr_model_path=detr_path,
            detr_processor_path=detr_processor_path,
            conf_threshold=conf_threshold
        )
        
        detector_evaluator = DetectorEvaluator(detector)
        results['detector'] = detector_evaluator.evaluate(
            annotation_file=test_annotation_file,
            images_dir=test_images_dir,
            max_images=max_images
        )
    else:
        print(f"⚠️ Файл аннотаций не найден: {test_annotation_file}")
    
    # 2. Оценка классификатора сцен
    print("\n\n[2/3] КЛАССИФИКАТОР СЦЕН")
    print("-" * 60)
    
    scene_test_path = Path(scene_test_dir)
    if scene_classifier_path and Path(scene_classifier_path).exists() and scene_test_path.exists():
        # Определяем тип классификатора
        if 'yolo' in scene_classifier_path.lower():
            from train_scene_yolo import SceneClassifierYOLO
            scene_classifier = SceneClassifierYOLO(scene_classifier_path)
        else:
            from train_scene_classifier import SceneClassifierInference
            scene_classifier = SceneClassifierInference(scene_classifier_path)
        
        scene_evaluator = SceneClassifierEvaluator(scene_classifier)
        results['scene_classifier'] = scene_evaluator.evaluate(
            test_dir=scene_test_dir,
            max_images=max_images
        )
    else:
        if not scene_classifier_path or not Path(scene_classifier_path).exists():
            print(f"⚠️ Классификатор сцен не найден")
            print(f"   Обучите: python train_scene_yolo.py --epochs 50")
        if not scene_test_path.exists():
            print(f"⚠️ Тестовые данные сцен не найдены: {scene_test_dir}")
    
    # 3. Оценка полной VLM
    print("\n\n[3/3] ПОЛНАЯ VLM СИСТЕМА")
    print("-" * 60)
    
    if Path(test_images_dir).exists():
        from VLM_Complete import CompleteVLM
        
        vlm = CompleteVLM(
            yolo_path=yolo_path,
            detr_path=detr_path,
            detr_processor_path=detr_processor_path,
            scene_classifier_path=scene_classifier_path,
            conf_threshold=conf_threshold
        )
        
        vlm_evaluator = VLMEvaluator(vlm)
        results['vlm'] = vlm_evaluator.evaluate(
            images_dir=test_images_dir,
            max_images=max_images
        )
    
    # Сохраняем результаты
    if output_file:
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\n✅ Результаты сохранены: {output_file}")
    
    # Финальная сводка
    print("\n" + "=" * 60)
    print("📊 ФИНАЛЬНАЯ СВОДКА")
    print("=" * 60)
    
    if 'detector' in results:
        d = results['detector']
        print(f"\n📦 Детектор мусора (ансамбль):")
        print(f"   • mAP:              {d['overall']['mAP']:.3f}")
        print(f"   • Precision:        {d['overall']['precision']:.3f}")
        print(f"   • Recall:           {d['overall']['recall']:.3f}")
        print(f"   • F1-score:         {d['overall']['f1']:.3f}")
        print(f"   • Inference time:   {d['timing']['avg_ms']:.1f} ms")
    
    if 'scene_classifier' in results and results['scene_classifier']:
        s = results['scene_classifier']
        print(f"\n🌍 Классификатор сцен (YOLO):")
        print(f"   • Accuracy:         {s['accuracy']:.3f}")
        print(f"   • Macro F1:         {s['macro_f1']:.3f}")
        print(f"   • Avg confidence:   {s['confidence']['mean']:.3f}")
        print(f"   • Inference time:   {s['timing']['avg_ms']:.1f} ms")
    
    if 'vlm' in results and results['vlm']:
        v = results['vlm']
        print(f"\n🤖 Полная VLM система:")
        print(f"   • Total detections: {sum(v['detection_stats'].values())}")
        print(f"   • Confident scenes: {v['confident_scenes_ratio']:.1%}")
        print(f"   • Inference time:   {v['timing']['avg_ms']:.1f} ms")
        print(f"   • FPS:              {v['timing']['fps']:.1f}")
    
    print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Оценка VLM системы")
    parser.add_argument("--yolo", type=str, default="models/yolo/yolov8x/best.pt")
    parser.add_argument("--detr", type=str, default="models/rt-detr/rt-detr-101/m")
    parser.add_argument("--detr_processor", type=str, default="models/rt-detr/rt-detr-101/p")
    parser.add_argument("--scene_classifier", type=str, default=None,
                        help="Путь к классификатору сцен (auto-detect если не указан)")
    parser.add_argument("--test_annotations", type=str, 
                        default="data/roboflow_dataset/test/_annotations.coco.json")
    parser.add_argument("--test_images", type=str, default="data/roboflow_dataset/test")
    parser.add_argument("--scene_test", type=str, default="data/scene_yolo_dataset/val")
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    
    args = parser.parse_args()
    
    evaluate_full_system(
        yolo_path=args.yolo,
        detr_path=args.detr,
        detr_processor_path=args.detr_processor,
        scene_classifier_path=args.scene_classifier,
        test_annotation_file=args.test_annotations,
        test_images_dir=args.test_images,
        scene_test_dir=args.scene_test,
        max_images=args.max_images,
        conf_threshold=args.conf_threshold,
        output_file=args.output
    )
