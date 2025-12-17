import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from transformers import T5ForConditionalGeneration, T5Tokenizer
from vlm_annotation.ensemble_detector import EnsembleDetector


GARBAGE_CLASSES = ['glass', 'metal', 'organic', 'paper', 'plastic']

PROMPT_TEMPLATE = """Describe the garbage detected based on this CV output:
{input}
Description:"""


def load_models(yolo_path, detr_path, detr_processor_path, scene_path, llm_path, conf_threshold):
    print("Loading models...")
    
    detector = EnsembleDetector(
        yolo_model_path=yolo_path,
        detr_model_path=detr_path,
        detr_processor_path=detr_processor_path,
        conf_threshold=conf_threshold
    )
    
    scene_classifier = None
    if scene_path and Path(scene_path).exists():
        if 'yolo' in scene_path.lower():
            from train_scene_yolo import SceneClassifierYOLO
            scene_classifier = SceneClassifierYOLO(scene_path)
        else:
            from train_scene_classifier import SceneClassifierInference
            scene_classifier = SceneClassifierInference(scene_path)
        print(f"Scene classifier loaded: {scene_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(llm_path)
    model = T5ForConditionalGeneration.from_pretrained(llm_path).to(device)
    model.eval()
    print(f"LLM loaded on {device}")
    
    return detector, scene_classifier, model, tokenizer, device


def get_cv_output(image, detector, scene_classifier):
    detections = detector.detect(image)
    
    detection_summary = [
        {"label": det["label"], "confidence": round(det["confidence"], 2)}
        for det in detections
    ]
    
    scene = {"class": "unknown", "confidence": 0.0}
    if scene_classifier:
        scene_result = scene_classifier.predict(image)
        scene = {
            "class": scene_result["class"],
            "confidence": round(scene_result["confidence"], 2)
        }
    
    return {"detections": detection_summary, "scene": scene}, detections


def generate_description(cv_output, model, tokenizer, device):
    prompt = PROMPT_TEMPLATE.format(input=json.dumps(cv_output))
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def check_factual_accuracy(description: str, cv_output: dict) -> dict:
    desc_lower = description.lower()
    
    counts = defaultdict(int)
    for det in cv_output["detections"]:
        counts[det["label"]] += 1
    
    results = {
        "class_mentioned": {},
        "count_correct": {},
        "scene_mentioned": False,
        "scene_correct": False,
        "no_garbage_correct": False,
    }
    
    total_detected = sum(counts.values())
    
    if total_detected == 0:
        no_garbage_phrases = ["no garbage", "no waste", "clean", "nothing detected"]
        results["no_garbage_correct"] = any(phrase in desc_lower for phrase in no_garbage_phrases)
    
    for cls in GARBAGE_CLASSES:
        actual_count = counts.get(cls, 0)
        
        cls_mentioned = cls in desc_lower
        results["class_mentioned"][cls] = cls_mentioned
        
        if actual_count > 0:
            if cls_mentioned:
                if str(actual_count) in description:
                    results["count_correct"][cls] = True
                else:
                    results["count_correct"][cls] = False
            else:
                results["count_correct"][cls] = False
        else:
            results["count_correct"][cls] = not cls_mentioned
    
    scene = cv_output["scene"]
    if scene["class"] != "unknown" and scene["confidence"] >= 0.8:
        results["scene_mentioned"] = scene["class"] in desc_lower
        results["scene_correct"] = results["scene_mentioned"]
    
    return results


def calculate_metrics(all_results: list) -> dict:
    metrics = {
        "total_samples": len(all_results),
        "class_mention_accuracy": {},
        "count_accuracy": {},
        "scene_accuracy": 0.0,
        "overall_factual_accuracy": 0.0,
    }
    
    class_mention_correct = defaultdict(int)
    class_mention_total = defaultdict(int)
    count_correct = defaultdict(int)
    count_total = defaultdict(int)
    scene_correct = 0
    scene_total = 0
    
    for result in all_results:
        fact_check = result["factual_check"]
        cv_output = result["cv_output"]
        
        counts = defaultdict(int)
        for det in cv_output["detections"]:
            counts[det["label"]] += 1
        
        for cls in GARBAGE_CLASSES:
            actual_count = counts.get(cls, 0)
            
            if actual_count > 0:
                class_mention_total[cls] += 1
                if fact_check["class_mentioned"].get(cls, False):
                    class_mention_correct[cls] += 1
                
                count_total[cls] += 1
                if fact_check["count_correct"].get(cls, False):
                    count_correct[cls] += 1
        
        if cv_output["scene"]["class"] != "unknown" and cv_output["scene"]["confidence"] >= 0.8:
            scene_total += 1
            if fact_check["scene_correct"]:
                scene_correct += 1
    
    for cls in GARBAGE_CLASSES:
        if class_mention_total[cls] > 0:
            metrics["class_mention_accuracy"][cls] = class_mention_correct[cls] / class_mention_total[cls]
        if count_total[cls] > 0:
            metrics["count_accuracy"][cls] = count_correct[cls] / count_total[cls]
    
    if scene_total > 0:
        metrics["scene_accuracy"] = scene_correct / scene_total
    
    all_correct = 0
    all_total = 0
    for cls in GARBAGE_CLASSES:
        all_correct += class_mention_correct[cls]
        all_total += class_mention_total[cls]
    all_correct += scene_correct
    all_total += scene_total
    
    if all_total > 0:
        metrics["overall_factual_accuracy"] = all_correct / all_total
    
    return metrics


def evaluate_llm(detector, scene_classifier, model, tokenizer, device, 
                 test_dir, max_images=None):
    print(f"\n{'='*60}")
    print("📝 EVALUATING DESCRIPTION LLM")
    print(f"{'='*60}")
    
    test_dir = Path(test_dir)
    image_paths = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if max_images:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, min(max_images, len(image_paths)))
    
    print(f"Testing on {len(image_paths)} images...")
    
    all_results = []
    description_lengths = []
    
    for img_path in tqdm(image_paths, desc="Evaluating"):
        try:
            image = Image.open(img_path).convert('RGB')
            
            cv_output, raw_detections = get_cv_output(image, detector, scene_classifier)
            
            description = generate_description(cv_output, model, tokenizer, device)
            
            fact_check = check_factual_accuracy(description, cv_output)
            
            all_results.append({
                "image": img_path.name,
                "cv_output": cv_output,
                "description": description,
                "factual_check": fact_check,
            })
            
            description_lengths.append(len(description.split()))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    metrics = calculate_metrics(all_results)
    
    metrics["avg_description_length"] = np.mean(description_lengths) if description_lengths else 0
    metrics["std_description_length"] = np.std(description_lengths) if description_lengths else 0
    
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📈 Class Mention Accuracy (when class is present):")
    for cls in GARBAGE_CLASSES:
        acc = metrics["class_mention_accuracy"].get(cls, 0)
        print(f"   {cls}: {acc:.1%}")
    
    print(f"\n📈 Count Accuracy (correct number mentioned):")
    for cls in GARBAGE_CLASSES:
        acc = metrics["count_accuracy"].get(cls, 0)
        print(f"   {cls}: {acc:.1%}")
    
    print(f"\n📈 Scene Accuracy: {metrics['scene_accuracy']:.1%}")
    print(f"📈 Overall Factual Accuracy: {metrics['overall_factual_accuracy']:.1%}")
    
    print(f"\n📈 Description Length: {metrics['avg_description_length']:.1f} ± {metrics['std_description_length']:.1f} words")
    
    print(f"\n📝 Sample Descriptions:")
    for i, result in enumerate(all_results[:5]):
        print(f"\n--- Sample {i+1}: {result['image']} ---")
        print(f"   CV: {len(result['cv_output']['detections'])} detections, scene={result['cv_output']['scene']['class']}")
        print(f"   LLM: {result['description']}")
    
    return metrics, all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Description LLM')
    parser.add_argument('--yolo', type=str, default="models/yolo/yolov8x/best.pt")
    parser.add_argument('--detr', type=str, default="models/rt-detr/rt-detr-101/m")
    parser.add_argument('--detr_processor', type=str, default="models/rt-detr/rt-detr-101/p")
    parser.add_argument('--scene', type=str, default="models/scene_classifier_yolo.pt")
    parser.add_argument('--llm', type=str, default="models/description_llm")
    parser.add_argument('--test_dir', type=str, default="data/1206-data/test")
    parser.add_argument('--max_images', type=int, default=100)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, default="llm_evaluation_results.json")
    args = parser.parse_args()
    
    if not Path(args.llm).exists():
        print(f"❌ LLM not found at {args.llm}")
        print("   Train it first: python train_description_llm.py")
        return
    
    detector, scene_classifier, model, tokenizer, device = load_models(
        args.yolo, args.detr, args.detr_processor, args.scene, args.llm, args.conf_threshold
    )
    
    metrics, results = evaluate_llm(
        detector, scene_classifier, model, tokenizer, device,
        args.test_dir, args.max_images
    )
    
    output_data = {
        "metrics": metrics,
        "samples": results[:20]
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()

