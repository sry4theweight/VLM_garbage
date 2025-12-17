import json
import random
import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

from vlm_annotation.ensemble_detector import EnsembleDetector


DESCRIPTION_TEMPLATES = [
    "There {verb} {garbage_desc}{scene_desc}.",
    "{garbage_desc} {verb} visible{scene_desc}.",
    "The image shows {garbage_desc}{scene_desc}.",
    "{garbage_desc} can be seen{scene_desc}.",
    "I can see {garbage_desc}{scene_desc}.",
    "Detected: {garbage_desc}{scene_desc}.",
    "Found {garbage_desc}{scene_desc}.",
    "This image contains {garbage_desc}{scene_desc}.",
    "{garbage_desc} detected{scene_desc}.",
    "The scene shows {garbage_desc}{scene_desc}.",
]

GARBAGE_PHRASES = {
    "plastic": ["plastic waste", "plastic items", "plastic debris", "plastic bottles", "plastic bags", "plastic containers", "plastic pieces"],
    "glass": ["glass bottles", "glass items", "broken glass", "glass waste", "glass shards", "glass containers", "glass debris"],
    "metal": ["metal cans", "metal waste", "metallic debris", "aluminum cans", "metal pieces", "tin cans", "metal scraps"],
    "paper": ["paper waste", "cardboard", "paper debris", "discarded paper", "cardboard pieces", "paper scraps", "cardboard boxes"],
    "organic": ["organic waste", "food scraps", "biodegradable waste", "organic matter", "food waste", "organic debris", "decomposing material"],
}

SCENE_PHRASES = {
    "grass": [" on the grass", " in a grassy area", " on green grass", " scattered on the lawn", " on grass surface", " in grass"],
    "sandy": [" on sandy ground", " on the beach", " in sandy terrain", " on the sand", " on sandy surface", " in sand"],
    "rocky": [" on rocky terrain", " among rocks", " on rocky ground", " between stones", " on rocks", " on rocky surface"],
    "marshy": [" in marshy area", " in wetlands", " in swampy terrain", " near marsh", " in wet area", " on marshy ground"],
}

PROMPT_TEMPLATE = """Describe the garbage detected based on this CV output:
{input}
Description:"""


def load_cv_models(yolo_path, detr_path, detr_processor_path, scene_path, conf_threshold):
    print("Loading CV models...")
    
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
    
    return detector, scene_classifier


def get_cv_output(image_path, detector, scene_classifier):
    image = Image.open(image_path).convert('RGB')
    
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
    
    return {"detections": detection_summary, "scene": scene}


def generate_description(cv_output):
    detections = cv_output["detections"]
    scene = cv_output["scene"]
    
    if not detections:
        if scene["class"] != "unknown" and scene["confidence"] >= 0.8:
            scene_phrase = random.choice(SCENE_PHRASES.get(scene["class"], [""]))
            return f"No garbage detected{scene_phrase}."
        return "No garbage detected in this image."
    
    counts = {}
    for det in detections:
        label = det["label"]
        counts[label] = counts.get(label, 0) + 1
    
    garbage_parts = []
    for label, count in counts.items():
        phrase = random.choice(GARBAGE_PHRASES.get(label, [label]))
        if count == 1:
            garbage_parts.append(f"1 {phrase.rstrip('s')}" if phrase.endswith('s') else f"1 {phrase}")
        else:
            garbage_parts.append(f"{count} {phrase}")
    
    if len(garbage_parts) == 1:
        garbage_desc = garbage_parts[0]
    elif len(garbage_parts) == 2:
        garbage_desc = f"{garbage_parts[0]} and {garbage_parts[1]}"
    else:
        garbage_desc = ", ".join(garbage_parts[:-1]) + f", and {garbage_parts[-1]}"
    
    scene_desc = ""
    if scene["class"] != "unknown" and scene["confidence"] >= 0.8:
        scene_desc = random.choice(SCENE_PHRASES.get(scene["class"], [""]))
    
    total = sum(counts.values())
    verb = "is" if total == 1 else "are"
    
    template = random.choice(DESCRIPTION_TEMPLATES)
    return template.format(verb=verb, garbage_desc=garbage_desc, scene_desc=scene_desc)


def create_training_dataset(image_dirs, detector, scene_classifier, max_images=None, variations=3):
    print("\n" + "="*50)
    print("Creating training dataset...")
    print("="*50)
    
    image_paths = []
    for img_dir in image_dirs:
        img_dir = Path(img_dir)
        if img_dir.exists():
            image_paths.extend(list(img_dir.glob("*.jpg")))
            image_paths.extend(list(img_dir.glob("*.png")))
    
    if max_images:
        random.shuffle(image_paths)
        image_paths = image_paths[:max_images]
    
    print(f"Processing {len(image_paths)} images with {variations} variations each...")
    
    dataset = []
    for img_path in tqdm(image_paths, desc="Generating data"):
        try:
            cv_output = get_cv_output(str(img_path), detector, scene_classifier)
            
            for _ in range(variations):
                description = generate_description(cv_output)
                if description and len(description.strip()) > 0:
                    dataset.append({
                        "input": json.dumps(cv_output),
                        "output": description,
                    })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Created {len(dataset)} training examples")
    return dataset


def preprocess_function(examples, tokenizer):
    inputs = [PROMPT_TEMPLATE.format(input=inp) for inp in examples["input"]]
    targets = examples["output"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=256, 
        truncation=True, 
        padding=False
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=128, 
            truncation=True, 
            padding=False
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_llm(dataset, output_dir, model_name, epochs, batch_size, learning_rate):
    print("\n" + "="*50)
    print("Training LLM...")
    print("="*50)
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    print(f"Model: {model_name}")
    print(f"Parameters: {model.num_parameters():,}")
    
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    print("\n--- Sample training data ---")
    for i in range(min(3, len(train_data))):
        print(f"Input: {train_data[i]['input'][:100]}...")
        print(f"Output: {train_data[i]['output']}")
        print()
    
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)
    
    train_ds = train_ds.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True, 
        remove_columns=train_ds.column_names
    )
    val_ds = val_ds.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True, 
        remove_columns=val_ds.column_names
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        save_total_limit=2,
        save_steps=200,
        logging_steps=50,
        predict_with_generate=True,
        fp16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    config = {
        "model_path": output_dir,
        "model_type": "t5",
        "base_model": model_name,
        "prompt_template": PROMPT_TEMPLATE,
        "max_input_length": 256,
        "max_output_length": 128,
        "num_beams": 4
    }
    
    with open(f"{output_dir}/llm_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to {output_dir}")
    return model, tokenizer


def test_model(model, tokenizer, detector, scene_classifier, test_dir, num_samples=5):
    print("\n" + "="*50)
    print("Testing model...")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    test_images = list(Path(test_dir).glob("*.jpg"))[:num_samples]
    
    for img_path in test_images:
        cv_output = get_cv_output(str(img_path), detector, scene_classifier)
        
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
        
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nImage: {img_path.name}")
        print(f"  Detections: {len(cv_output['detections'])}, Scene: {cv_output['scene']['class']}")
        print(f"  LLM: {description}")


def main():
    parser = argparse.ArgumentParser(description='Train Description LLM')
    parser.add_argument('--yolo', type=str, default="models/yolo/yolov8x/best.pt")
    parser.add_argument('--detr', type=str, default="models/rt-detr/rt-detr-101/m")
    parser.add_argument('--detr_processor', type=str, default="models/rt-detr/rt-detr-101/p")
    parser.add_argument('--scene', type=str, default="models/scene_classifier_yolo.pt")
    parser.add_argument('--train_dirs', type=str, nargs='+', 
                       default=["data/1206-data/train", "data/1206-data/valid"])
    parser.add_argument('--test_dir', type=str, default="data/1206-data/test")
    parser.add_argument('--output', type=str, default="models/description_llm")
    parser.add_argument('--model', type=str, default="google/flan-t5-base",
                       help="Base model (flan-t5-small, flan-t5-base, flan-t5-large)")
    parser.add_argument('--max_images', type=int, default=3000)
    parser.add_argument('--variations', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--skip_train', action='store_true', help="Skip training, only test")
    args = parser.parse_args()
    
    print("="*60)
    print("DESCRIPTION LLM TRAINING")
    print("="*60)
    print(f"Base model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Max images: {args.max_images}")
    print(f"Variations: {args.variations}")
    print(f"Epochs: {args.epochs}")
    
    detector, scene_classifier = load_cv_models(
        args.yolo, args.detr, args.detr_processor, args.scene, args.conf_threshold
    )
    
    if not args.skip_train:
        dataset = create_training_dataset(
            args.train_dirs, detector, scene_classifier, 
            args.max_images, args.variations
        )
        
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with open(f"{args.output}/training_data.json", 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Training data saved to {args.output}/training_data.json")
        
        model, tokenizer = train_llm(
            dataset, args.output, args.model, 
            args.epochs, args.batch_size, args.lr
        )
    else:
        print("\nLoading existing model...")
        tokenizer = T5Tokenizer.from_pretrained(args.output)
        model = T5ForConditionalGeneration.from_pretrained(args.output)
    
    test_model(model, tokenizer, detector, scene_classifier, args.test_dir)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()

