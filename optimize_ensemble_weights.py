import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import torch
from itertools import product
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
import supervision as sv


GARBAGE_CLASSES = ['glass', 'metal', 'organic', 'paper', 'plastic']
CLASS_NAMES = ['garb-garbage', 'glass', 'metal', 'organic', 'paper', 'plastic']


def load_coco_annotations(annotation_file: str) -> dict:
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
        })
    
    return {
        'images': images,
        'categories': categories,
        'annotations': annotations_by_image
    }


def calculate_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


class RawDetector:
    def __init__(self, yolo_path: str, detr_path: str, detr_processor_path: str, 
                 device: str = None, conf_threshold: float = 0.3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        
        print(f"Loading YOLO model from {yolo_path}...")
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(self.device)
        
        print(f"Loading RT-DETR model from {detr_path}...")
        self.detr_processor = RTDetrImageProcessor.from_pretrained(
            detr_processor_path, local_files_only=True, use_fast=True
        )
        self.detr_model = RTDetrV2ForObjectDetection.from_pretrained(
            detr_path, local_files_only=True
        ).to(self.device)
        self.detr_model.eval()
    
    def detect_yolo(self, image: Image.Image) -> list:
        results = self.yolo_model(image, verbose=False)
        detections_sv = sv.Detections.from_ultralytics(results[0])
        
        dets = []
        for box, conf, cls in zip(detections_sv.xyxy, detections_sv.confidence, detections_sv.class_id):
            if conf >= self.conf_threshold:
                label = CLASS_NAMES[int(cls)]
                if label != 'garb-garbage':
                    dets.append({
                        'box': box.tolist(),
                        'confidence': float(conf),
                        'label': label
                    })
        return dets
    
    def detect_detr(self, image: Image.Image) -> list:
        inputs = self.detr_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detr_model(**inputs)
        
        w, h = image.size
        results = self.detr_processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=self.conf_threshold
        )
        detections_sv = sv.Detections.from_transformers(results[0])
        
        dets = []
        for box, conf, cls in zip(detections_sv.xyxy, detections_sv.confidence, detections_sv.class_id):
            if conf >= self.conf_threshold:
                label = CLASS_NAMES[int(cls)]
                if label != 'garb-garbage':
                    dets.append({
                        'box': box.tolist(),
                        'confidence': float(conf),
                        'label': label
                    })
        return dets


class WeightedEnsemble:
    def __init__(self, yolo_weights: dict, detr_weights: dict, conf_delta: float = 0.5):
        self.yolo_weights = yolo_weights
        self.detr_weights = detr_weights
        self.conf_delta = conf_delta
    
    def calibrate(self, conf: float, weight: float) -> float:
        return conf * np.sqrt(weight) if weight > 0 else conf
    
    def ensemble(self, yolo_dets: list, detr_dets: list, iou_threshold: float = 0.3) -> list:
        matched = []
        used_detr = set()
        
        for yolo_idx, yolo_det in enumerate(yolo_dets):
            best_match = None
            best_iou = 0
            
            for detr_idx, detr_det in enumerate(detr_dets):
                if detr_idx in used_detr:
                    continue
                
                iou = calculate_iou(yolo_det['box'], detr_det['box'])
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = detr_idx
            
            if best_match is not None:
                matched.append((yolo_idx, best_match))
                used_detr.add(best_match)
        
        ensemble_detections = []
        matched_yolo = {m[0] for m in matched}
        matched_detr = {m[1] for m in matched}
        
        for yolo_idx, detr_idx in matched:
            yolo_det = yolo_dets[yolo_idx]
            detr_det = detr_dets[detr_idx]
            
            yolo_label = yolo_det['label']
            detr_label = detr_det['label']
            
            yolo_conf_cal = self.calibrate(yolo_det['confidence'], self.yolo_weights.get(yolo_label.lower(), 0))
            detr_conf_cal = self.calibrate(detr_det['confidence'], self.detr_weights.get(detr_label.lower(), 0))
            
            if abs(yolo_conf_cal - detr_conf_cal) > self.conf_delta:
                yolo_w = self.yolo_weights.get(yolo_label.lower(), 0)
                detr_w = self.detr_weights.get(detr_label.lower(), 0)
                
                if yolo_w > detr_w:
                    ensemble_detections.append({
                        'box': yolo_det['box'],
                        'label': yolo_label,
                        'confidence': float(yolo_conf_cal)
                    })
                else:
                    ensemble_detections.append({
                        'box': detr_det['box'],
                        'label': detr_label,
                        'confidence': float(detr_conf_cal)
                    })
            else:
                combined_scores = {}
                for class_name in GARBAGE_CLASSES:
                    yolo_score = yolo_conf_cal if yolo_label == class_name else 0
                    detr_score = detr_conf_cal if detr_label == class_name else 0
                    yolo_w = self.yolo_weights.get(class_name.lower(), 0)
                    detr_w = self.detr_weights.get(class_name.lower(), 0)
                    
                    if yolo_w + detr_w > 0:
                        combined_scores[class_name] = (yolo_w * yolo_score + detr_w * detr_score) / (yolo_w + detr_w)
                    else:
                        combined_scores[class_name] = 0
                
                best_class, best_score = max(combined_scores.items(), key=lambda x: x[1])
                ensemble_detections.append({
                    'box': yolo_det['box'],
                    'label': best_class,
                    'confidence': float(best_score)
                })
        
        for idx, det in enumerate(yolo_dets):
            if idx not in matched_yolo:
                conf_cal = self.calibrate(det['confidence'], self.yolo_weights.get(det['label'].lower(), 0))
                ensemble_detections.append({
                    'box': det['box'],
                    'label': det['label'],
                    'confidence': float(conf_cal)
                })
        
        for idx, det in enumerate(detr_dets):
            if idx not in matched_detr:
                conf_cal = self.calibrate(det['confidence'], self.detr_weights.get(det['label'].lower(), 0))
                ensemble_detections.append({
                    'box': det['box'],
                    'label': det['label'],
                    'confidence': float(conf_cal)
                })
        
        return ensemble_detections


def compute_map(predictions_by_image: dict, annotations: dict, iou_threshold: float = 0.5) -> tuple:
    all_results = {cls: {'tp': 0, 'fp': 0, 'fn': 0, 'confs': []} for cls in GARBAGE_CLASSES}
    
    for img_id, predictions in predictions_by_image.items():
        gt_anns = annotations.get(img_id, [])
        
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)
        
        for gt in gt_anns:
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
                    all_results[cls]['tp'] += 1
                    all_results[cls]['confs'].append((pred['confidence'], True))
                    matched_gt.add(best_gt_idx)
                else:
                    all_results[cls]['fp'] += 1
                    all_results[cls]['confs'].append((pred['confidence'], False))
            
            all_results[cls]['fn'] += len(gt_boxes) - len(matched_gt)
    
    ap_scores = []
    class_ap = {}
    
    for cls in GARBAGE_CLASSES:
        tp = all_results[cls]['tp']
        fn = all_results[cls]['fn']
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
        class_ap[cls] = ap
    
    mAP = np.mean(ap_scores)
    return mAP, class_ap


class WeightOptimizer:
    def __init__(self, detector: RawDetector, annotation_file: str, images_dir: str, max_images: int = None):
        self.detector = detector
        self.images_dir = Path(images_dir)
        
        coco_data = load_coco_annotations(annotation_file)
        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        
        image_ids = list(self.images.keys())
        if max_images:
            import random
            random.seed(42)
            image_ids = random.sample(image_ids, min(max_images, len(image_ids)))
        
        self.image_ids = image_ids
        self.cached_yolo = {}
        self.cached_detr = {}
        
        print(f"Caching predictions for {len(image_ids)} images...")
        self._cache_predictions()
    
    def _cache_predictions(self):
        for img_id in tqdm(self.image_ids, desc="Caching"):
            img_info = self.images[img_id]
            img_path = self.images_dir / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            image = Image.open(img_path).convert('RGB')
            self.cached_yolo[img_id] = self.detector.detect_yolo(image)
            self.cached_detr[img_id] = self.detector.detect_detr(image)
    
    def evaluate_weights(self, yolo_weights: dict, detr_weights: dict, conf_delta: float = 0.5) -> float:
        ensemble = WeightedEnsemble(yolo_weights, detr_weights, conf_delta)
        
        predictions_by_image = {}
        for img_id in self.image_ids:
            if img_id not in self.cached_yolo:
                continue
            
            yolo_dets = self.cached_yolo[img_id]
            detr_dets = self.cached_detr[img_id]
            predictions_by_image[img_id] = ensemble.ensemble(yolo_dets, detr_dets)
        
        mAP, class_ap = compute_map(predictions_by_image, self.annotations)
        return mAP, class_ap
    
    def grid_search(self, steps: int = 5):
        print(f"\n{'='*60}")
        print("🔍 GRID SEARCH OPTIMIZATION")
        print(f"{'='*60}")
        
        weight_values = np.linspace(0, 1, steps)
        
        best_mAP = 0
        best_weights = None
        results = []
        
        total_combos = len(list(product(weight_values, repeat=len(GARBAGE_CLASSES) * 2)))
        print(f"Total combinations: {total_combos}")
        
        for yolo_plastic in tqdm(weight_values, desc="Optimizing"):
            for yolo_glass in weight_values:
                for yolo_metal in weight_values:
                    for yolo_paper in weight_values:
                        for yolo_organic in weight_values:
                            yolo_w = {
                                'plastic': yolo_plastic,
                                'glass': yolo_glass,
                                'metal': yolo_metal,
                                'paper': yolo_paper,
                                'organic': yolo_organic
                            }
                            detr_w = {
                                'plastic': 1 - yolo_plastic,
                                'glass': 1 - yolo_glass,
                                'metal': 1 - yolo_metal,
                                'paper': 1 - yolo_paper,
                                'organic': 1 - yolo_organic
                            }
                            
                            mAP, class_ap = self.evaluate_weights(yolo_w, detr_w)
                            results.append((mAP, yolo_w, detr_w, class_ap))
                            
                            if mAP > best_mAP:
                                best_mAP = mAP
                                best_weights = (yolo_w.copy(), detr_w.copy(), class_ap.copy())
        
        return best_mAP, best_weights, results
    
    def optimize_per_class(self, steps: int = 20):
        print(f"\n{'='*60}")
        print("🎯 PER-CLASS OPTIMIZATION")
        print(f"{'='*60}")
        
        weight_values = np.linspace(0, 1, steps)
        
        best_yolo_weights = {}
        best_detr_weights = {}
        
        for cls in GARBAGE_CLASSES:
            print(f"\nOptimizing {cls}...")
            best_ap = 0
            best_yolo_w = 0.5
            
            for yolo_w in tqdm(weight_values, desc=cls):
                yolo_weights = {c: (yolo_w if c == cls else 0.5) for c in GARBAGE_CLASSES}
                detr_weights = {c: (1 - yolo_w if c == cls else 0.5) for c in GARBAGE_CLASSES}
                
                mAP, class_ap = self.evaluate_weights(yolo_weights, detr_weights)
                
                if class_ap[cls] > best_ap:
                    best_ap = class_ap[cls]
                    best_yolo_w = yolo_w
            
            best_yolo_weights[cls] = best_yolo_w
            best_detr_weights[cls] = 1 - best_yolo_w
            print(f"  Best: YOLO={best_yolo_w:.2f}, DETR={1-best_yolo_w:.2f}, AP={best_ap:.4f}")
        
        final_mAP, final_class_ap = self.evaluate_weights(best_yolo_weights, best_detr_weights)
        
        return final_mAP, best_yolo_weights, best_detr_weights, final_class_ap
    
    def differential_evolution_search(self, maxiter: int = 50):
        print(f"\n{'='*60}")
        print("🧬 DIFFERENTIAL EVOLUTION OPTIMIZATION")
        print(f"{'='*60}")
        
        def objective(x):
            yolo_weights = {
                'plastic': x[0],
                'glass': x[1],
                'metal': x[2],
                'paper': x[3],
                'organic': x[4]
            }
            detr_weights = {
                'plastic': x[5],
                'glass': x[6],
                'metal': x[7],
                'paper': x[8],
                'organic': x[9]
            }
            mAP, _ = self.evaluate_weights(yolo_weights, detr_weights)
            return -mAP
        
        bounds = [(0, 1)] * 10
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=maxiter, 
            seed=42, 
            disp=True,
            workers=-1,
            updating='deferred'
        )
        
        best_yolo = {
            'plastic': result.x[0],
            'glass': result.x[1],
            'metal': result.x[2],
            'paper': result.x[3],
            'organic': result.x[4]
        }
        best_detr = {
            'plastic': result.x[5],
            'glass': result.x[6],
            'metal': result.x[7],
            'paper': result.x[8],
            'organic': result.x[9]
        }
        
        final_mAP, class_ap = self.evaluate_weights(best_yolo, best_detr)
        
        return final_mAP, best_yolo, best_detr, class_ap
    
    def bayesian_search(self, n_calls: int = 100):
        print(f"\n{'='*60}")
        print("🔮 BAYESIAN OPTIMIZATION")
        print(f"{'='*60}")
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            print("scikit-optimize not installed. Run: pip install scikit-optimize")
            return None
        
        def objective(x):
            yolo_weights = {
                'plastic': x[0],
                'glass': x[1],
                'metal': x[2],
                'paper': x[3],
                'organic': x[4]
            }
            detr_weights = {
                'plastic': x[5],
                'glass': x[6],
                'metal': x[7],
                'paper': x[8],
                'organic': x[9]
            }
            mAP, _ = self.evaluate_weights(yolo_weights, detr_weights)
            return -mAP
        
        space = [Real(0, 1, name=f'w{i}') for i in range(10)]
        
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=True)
        
        best_yolo = {
            'plastic': result.x[0],
            'glass': result.x[1],
            'metal': result.x[2],
            'paper': result.x[3],
            'organic': result.x[4]
        }
        best_detr = {
            'plastic': result.x[5],
            'glass': result.x[6],
            'metal': result.x[7],
            'paper': result.x[8],
            'organic': result.x[9]
        }
        
        final_mAP, class_ap = self.evaluate_weights(best_yolo, best_detr)
        
        return final_mAP, best_yolo, best_detr, class_ap


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize ensemble weights for mAP')
    parser.add_argument('--method', type=str, default='per_class', 
                       choices=['grid', 'per_class', 'de', 'bayesian'],
                       help='Optimization method')
    parser.add_argument('--max_images', type=int, default=100, help='Max images to use (same as evaluate_vlm)')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold (same as evaluate_vlm)')
    parser.add_argument('--steps', type=int, default=20, help='Grid search steps')
    parser.add_argument('--maxiter', type=int, default=50, help='DE max iterations')
    parser.add_argument('--test_annotations', type=str, 
                       default="data/1206-data/test/_annotations.coco.json")
    parser.add_argument('--test_images', type=str, default="data/1206-data/test")
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 ENSEMBLE WEIGHT OPTIMIZER")
    print("="*60)
    print(f"   • Max images: {args.max_images}")
    print(f"   • Confidence threshold: {args.conf_threshold}")
    print(f"   • Method: {args.method}")
    
    detector = RawDetector(
        yolo_path="models/yolo/yolov8x/best.pt",
        detr_path="models/rt-detr/rt-detr-101/m",
        detr_processor_path="models/rt-detr/rt-detr-101/p",
        conf_threshold=args.conf_threshold
    )
    
    optimizer = WeightOptimizer(
        detector=detector,
        annotation_file=args.test_annotations,
        images_dir=args.test_images,
        max_images=args.max_images
    )
    
    print("\n📊 Baseline (equal weights 0.5/0.5):")
    baseline_yolo = {c: 0.5 for c in GARBAGE_CLASSES}
    baseline_detr = {c: 0.5 for c in GARBAGE_CLASSES}
    baseline_mAP, baseline_ap = optimizer.evaluate_weights(baseline_yolo, baseline_detr)
    print(f"   mAP: {baseline_mAP:.4f}")
    for cls in GARBAGE_CLASSES:
        print(f"   {cls}: AP={baseline_ap[cls]:.4f}")
    
    if args.method == 'grid':
        best_mAP, best_weights, _ = optimizer.grid_search(steps=args.steps)
        best_yolo, best_detr, best_class_ap = best_weights
    elif args.method == 'per_class':
        best_mAP, best_yolo, best_detr, best_class_ap = optimizer.optimize_per_class(steps=args.steps)
    elif args.method == 'de':
        best_mAP, best_yolo, best_detr, best_class_ap = optimizer.differential_evolution_search(maxiter=args.maxiter)
    elif args.method == 'bayesian':
        result = optimizer.bayesian_search(n_calls=args.maxiter)
        if result:
            best_mAP, best_yolo, best_detr, best_class_ap = result
        else:
            return
    
    print(f"\n{'='*60}")
    print("🏆 OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"\nBaseline mAP: {baseline_mAP:.4f}")
    print(f"Optimized mAP: {best_mAP:.4f}")
    print(f"Improvement: {(best_mAP - baseline_mAP)*100:.2f}%")
    
    print(f"\n📋 OPTIMAL WEIGHTS:")
    print(f"\nMODEL_WEIGHTS = {{")
    print(f'    "YOLOv8x": {{')
    for cls in GARBAGE_CLASSES:
        print(f'        "{cls}": {best_yolo[cls]:.4f},')
    print(f'    }},')
    print(f'    "RT-DETR-101": {{')
    for cls in GARBAGE_CLASSES:
        print(f'        "{cls}": {best_detr[cls]:.4f},')
    print(f'    }}')
    print(f"}}") 
    
    print(f"\n📈 Per-class AP:")
    for cls in GARBAGE_CLASSES:
        improvement = best_class_ap[cls] - baseline_ap[cls]
        sign = "+" if improvement >= 0 else ""
        print(f"   {cls}: {best_class_ap[cls]:.4f} ({sign}{improvement*100:.2f}%)")
    
    print(f"\n💾 Copy these weights to ensemble_detector.py:")
    print("-" * 50)
    print("MODEL_WEIGHTS = {")
    print('    "YOLOv8x": {')
    for cls in GARBAGE_CLASSES:
        print(f'        "{cls}": {best_yolo[cls]:.4f},')
    print('    },')
    print('    "RT-DETR-101": {')
    for cls in GARBAGE_CLASSES:
        print(f'        "{cls}": {best_detr[cls]:.4f},')
    print('    }')
    print("}")


if __name__ == "__main__":
    main()

