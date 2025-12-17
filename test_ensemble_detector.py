"""
Тестирование ансамбля детекторов (YOLO + RT-DETR) на тестовом датасете Roboflow

Метрики:
- mAP@50 - mean Average Precision при IoU=0.5
- mAP@50:95 - mean Average Precision, усреднённое по IoU от 0.5 до 0.95
- mAR@50 - mean Average Recall при IoU=0.5

Запуск:
    python test_ensemble_detector.py
    python test_ensemble_detector.py --conf 0.5
    python test_ensemble_detector.py --split valid
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm


# Классы мусора (без garb-garbage)
GARBAGE_CLASSES = ['glass', 'metal', 'organic', 'paper', 'plastic']


def load_coco_annotations(annotation_file: str) -> dict:
    """Загрузка COCO аннотаций"""
    print(f"📂 Загрузка аннотаций: {annotation_file}")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    
    print(f"   Категории: {list(categories.values())}")
    
    # Группируем аннотации по изображениям
    annotations_by_image = defaultdict(list)
    skipped = 0
    
    for ann in coco['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height] в COCO формате
        
        # Конвертируем в [x1, y1, x2, y2]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        category_name = categories.get(category_id, 'unknown')
        
        # Пропускаем garb-garbage
        if category_name == 'garb-garbage' or category_name not in GARBAGE_CLASSES:
            skipped += 1
            continue
            
        annotations_by_image[image_id].append({
            'bbox': [x1, y1, x2, y2],
            'label': category_name
        })
    
    total_objects = sum(len(anns) for anns in annotations_by_image.values())
    print(f"   Изображений: {len(images)}")
    print(f"   Объектов: {total_objects} (пропущено garb-garbage: {skipped})")
    
    return {
        'images': images,
        'categories': categories,
        'annotations': annotations_by_image
    }


def calculate_iou(box1: list, box2: list) -> float:
    """IoU между двумя bbox [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def compute_ap(recalls, precisions):
    """Вычисление AP через интерполяцию (COCO style)"""
    # Добавляем начальную и конечную точки
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # Интерполяция precision (убывающая огибающая)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Находим точки где recall меняется
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    
    # Вычисляем AP как площадь под кривой
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap


def evaluate_at_iou(all_predictions: dict, all_gt: dict, iou_threshold: float):
    """
    Оценка при заданном IoU threshold
    
    Returns:
        dict с AP и AR для каждого класса
    """
    results = {}
    
    for cls in GARBAGE_CLASSES:
        # Собираем все предсказания и GT для класса
        all_preds = []
        all_gts = []
        
        for img_id in all_predictions.keys():
            preds = [p for p in all_predictions[img_id] if p['label'] == cls]
            gts = [g for g in all_gt.get(img_id, []) if g['label'] == cls]
            
            for p in preds:
                all_preds.append({
                    'confidence': p['confidence'],
                    'box': p['box'],
                    'img_id': img_id
                })
            
            for g in gts:
                all_gts.append({
                    'bbox': g['bbox'],
                    'img_id': img_id,
                    'matched': False
                })
        
        if len(all_gts) == 0:
            results[cls] = {'ap': 0.0, 'recall': 0.0, 'precision': 0.0}
            continue
        
        # Сортируем предсказания по confidence
        all_preds = sorted(all_preds, key=lambda x: -x['confidence'])
        
        # Считаем TP/FP для каждого предсказания
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        
        # Отслеживаем matched GT по изображениям
        gt_matched = {img_id: set() for img_id in all_predictions.keys()}
        
        for pred_idx, pred in enumerate(all_preds):
            img_id = pred['img_id']
            
            # Находим GT для этого изображения
            img_gts = [(i, g) for i, g in enumerate(all_gts) 
                      if g['img_id'] == img_id and i not in gt_matched[img_id]]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in img_gts:
                iou = calculate_iou(pred['box'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                gt_matched[img_id].add(best_gt_idx)
            else:
                fp[pred_idx] = 1
        
        # Кумулятивные суммы
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision и Recall
        recalls = tp_cumsum / len(all_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # AP
        ap = compute_ap(recalls, precisions)
        
        # Максимальный recall (для AR)
        max_recall = recalls[-1] if len(recalls) > 0 else 0
        
        # Precision при максимальном recall
        final_precision = precisions[-1] if len(precisions) > 0 else 0
        
        results[cls] = {
            'ap': ap,
            'recall': max_recall,
            'precision': final_precision,
            'tp': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
            'fp': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0,
            'fn': len(all_gts) - (int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0),
            'total_gt': len(all_gts),
            'total_pred': len(all_preds)
        }
    
    return results


def test_ensemble(
    dataset_dir: str = "data/complete_dataset",
    split: str = "test",
    yolo_path: str = "models/yolo/yolov8x/best.pt",
    detr_path: str = "models/rt-detr/rt-detr-101/m",
    detr_processor_path: str = "models/rt-detr/rt-detr-101/p",
    conf_threshold: float = 0.01,  # Низкий порог для правильного расчёта AP
    save_results: bool = True
):
    """
    Тестирование ансамбля на датасете
    
    Метрики:
    - mAP@50: mean Average Precision при IoU=0.5
    - mAP@50:95: mean AP, усреднённое по IoU от 0.5 до 0.95 с шагом 0.05
    - mAR@50: mean Average Recall при IoU=0.5
    """
    print("\n" + "=" * 70)
    print("🔬 ТЕСТИРОВАНИЕ АНСАМБЛЯ ДЕТЕКТОРОВ")
    print("=" * 70)
    
    # Пути
    dataset_path = Path(dataset_dir)
    split_dir = dataset_path / split
    annotation_file = split_dir / "_annotations.coco.json"
    
    if not split_dir.exists():
        print(f"❌ Папка не найдена: {split_dir}")
        return None
    
    if not annotation_file.exists():
        print(f"❌ Файл аннотаций не найден: {annotation_file}")
        return None
    
    print(f"\n📁 Датасет: {dataset_dir}")
    print(f"📁 Split: {split}")
    print(f"📁 Confidence threshold: {conf_threshold}")
    
    # Загружаем аннотации
    coco_data = load_coco_annotations(str(annotation_file))
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # Загружаем ансамбль
    print(f"\n🔧 Загрузка ансамбля...")
    print(f"   YOLO: {yolo_path}")
    print(f"   RT-DETR: {detr_path}")
    
    from vlm_annotation.ensemble_detector import EnsembleDetector
    
    detector = EnsembleDetector(
        yolo_model_path=yolo_path,
        detr_model_path=detr_path,
        detr_processor_path=detr_processor_path,
        conf_threshold=conf_threshold
    )
    print("✅ Ансамбль загружен!")
    
    # Собираем все предсказания
    print(f"\n🚀 Запуск инференса на {len(images)} изображениях...")
    
    all_predictions = {}
    all_gt = {}
    inference_times = []
    processed = 0
    
    for img_id in tqdm(images.keys(), desc="Инференс"):
        img_info = images[img_id]
        img_path = split_dir / img_info['file_name']
        
        if not img_path.exists():
            continue
        
        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')
        
        # Детекция
        start_time = time.time()
        predictions = detector.detect(image)
        inference_times.append(time.time() - start_time)
        
        all_predictions[img_id] = predictions
        all_gt[img_id] = annotations.get(img_id, [])
        processed += 1
    
    # Вычисляем метрики при разных IoU
    print(f"\n📊 Вычисление метрик...")
    
    # IoU thresholds для mAP@50:95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, 0.6, ..., 0.95]
    
    # mAP@50
    results_50 = evaluate_at_iou(all_predictions, all_gt, 0.5)
    
    # mAP при каждом IoU для mAP@50:95
    ap_per_iou = []
    for iou_th in iou_thresholds:
        results_iou = evaluate_at_iou(all_predictions, all_gt, iou_th)
        mean_ap = np.mean([r['ap'] for r in results_iou.values()])
        ap_per_iou.append(mean_ap)
    
    # Итоговые метрики
    mAP50 = np.mean([r['ap'] for r in results_50.values()])
    mAP50_95 = np.mean(ap_per_iou)
    mAR50 = np.mean([r['recall'] for r in results_50.values()])
    
    # Вывод результатов
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ПО КЛАССАМ (IoU=0.5)")
    print("=" * 70)
    
    print(f"\n{'Класс':<12} {'AP@50':>10} {'Recall':>10} {'Precision':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
    print("-" * 70)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt, total_pred = 0, 0
    
    for cls in GARBAGE_CLASSES:
        r = results_50[cls]
        print(f"{cls:<12} {r['ap']:>10.4f} {r['recall']:>10.4f} {r['precision']:>10.4f} {r['tp']:>8} {r['fp']:>8} {r['fn']:>8}")
        total_tp += r['tp']
        total_fp += r['fp']
        total_fn += r['fn']
        total_gt += r['total_gt']
        total_pred += r['total_pred']
    
    print("-" * 70)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"{'OVERALL':<12} {mAP50:>10.4f} {overall_recall:>10.4f} {overall_precision:>10.4f} {total_tp:>8} {total_fp:>8} {total_fn:>8}")
    
    # mAP@50:95 по IoU
    print(f"\n{'='*70}")
    print("📊 AP ПО РАЗНЫМ IoU THRESHOLDS")
    print("=" * 70)
    print(f"\n{'IoU':>8}", end="")
    for cls in GARBAGE_CLASSES:
        print(f"{cls[:6]:>10}", end="")
    print(f"{'mAP':>10}")
    print("-" * 70)
    
    for i, iou_th in enumerate(iou_thresholds):
        results_iou = evaluate_at_iou(all_predictions, all_gt, iou_th)
        print(f"{iou_th:>8.2f}", end="")
        for cls in GARBAGE_CLASSES:
            print(f"{results_iou[cls]['ap']:>10.4f}", end="")
        print(f"{ap_per_iou[i]:>10.4f}")
    
    # Главные метрики
    print("\n" + "=" * 70)
    print("🏆 ГЛАВНЫЕ МЕТРИКИ")
    print("=" * 70)
    print(f"\n   📊 mAP@50:        {mAP50:.4f}")
    print(f"   📊 mAP@50:95:     {mAP50_95:.4f}")
    print(f"   📊 mAR@50:        {mAR50:.4f}")
    print()
    print(f"   📈 Precision@50:  {overall_precision:.4f}")
    print(f"   📈 Recall@50:     {overall_recall:.4f}")
    print(f"   📈 F1@50:         {2*overall_precision*overall_recall/(overall_precision+overall_recall) if (overall_precision+overall_recall) > 0 else 0:.4f}")
    
    # Статистика
    print(f"\n📈 СТАТИСТИКА")
    print(f"   • Изображений:    {processed}")
    print(f"   • GT объектов:    {total_gt}")
    print(f"   • Предсказаний:   {total_pred}")
    print(f"   • True Positives: {total_tp}")
    print(f"   • False Positives:{total_fp}")
    print(f"   • False Negatives:{total_fn}")
    
    # Время
    print(f"\n⏱️ ВРЕМЯ ИНФЕРЕНСА")
    print(f"   • Среднее:        {np.mean(inference_times)*1000:.1f} ms")
    print(f"   • Медиана:        {np.median(inference_times)*1000:.1f} ms")
    print(f"   • FPS:            {1/np.mean(inference_times):.1f}")
    
    # Сохраняем результаты
    if save_results:
        results_data = {
            'config': {
                'dataset': dataset_dir,
                'split': split,
                'conf_threshold': conf_threshold,
                'yolo_path': yolo_path,
                'detr_path': detr_path
            },
            'metrics': {
                'mAP50': float(mAP50),
                'mAP50_95': float(mAP50_95),
                'mAR50': float(mAR50),
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1': float(2*overall_precision*overall_recall/(overall_precision+overall_recall) if (overall_precision+overall_recall) > 0 else 0)
            },
            'per_class_AP50': {cls: float(results_50[cls]['ap']) for cls in GARBAGE_CLASSES},
            'per_class_AR50': {cls: float(results_50[cls]['recall']) for cls in GARBAGE_CLASSES},
            'ap_per_iou': {f"IoU_{iou:.2f}": float(ap) for iou, ap in zip(iou_thresholds, ap_per_iou)},
            'counts': {
                'images': processed,
                'total_gt': total_gt,
                'total_pred': total_pred,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn
            },
            'timing': {
                'avg_ms': float(np.mean(inference_times) * 1000),
                'median_ms': float(np.median(inference_times) * 1000),
                'fps': float(1 / np.mean(inference_times))
            }
        }
        
        output_file = f"ensemble_test_results_{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Результаты сохранены: {output_file}")
    
    print("\n" + "=" * 70)
    
    return {
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'mAR50': mAR50
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирование ансамбля детекторов")
    parser.add_argument("--dataset", type=str, default="data/complete_dataset",
                        help="Путь к датасету")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"],
                        help="Какой split тестировать")
    parser.add_argument("--yolo", type=str, default="models/yolo/yolov8x/best.pt",
                        help="Путь к YOLO модели")
    parser.add_argument("--detr", type=str, default="models/rt-detr/rt-detr-101/m",
                        help="Путь к RT-DETR модели")
    parser.add_argument("--detr_processor", type=str, default="models/rt-detr/rt-detr-101/p",
                        help="Путь к RT-DETR процессору")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="Порог уверенности (низкий для правильного AP)")
    parser.add_argument("--no_save", action="store_true",
                        help="Не сохранять результаты")
    
    args = parser.parse_args()
    
    test_ensemble(
        dataset_dir=args.dataset,
        split=args.split,
        yolo_path=args.yolo,
        detr_path=args.detr,
        detr_processor_path=args.detr_processor,
        conf_threshold=args.conf,
        save_results=not args.no_save
    )
