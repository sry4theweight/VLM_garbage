import json
import random
import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

from vlm_annotation.ensemble_detector import EnsembleDetector


SYSTEM_PROMPT = """You are a helpful assistant that analyzes garbage detection results. 
You receive structured data about detected garbage objects and scene information from computer vision models.
Provide natural, informative responses about what garbage is present, where it is located, and answer questions naturally."""

CONVERSATION_TEMPLATES = [
    {
        "user": "What do you see in this image?",
        "assistant_template": "{description}"
    },
    {
        "user": "Describe the garbage in this image.",
        "assistant_template": "{description}"
    },
    {
        "user": "Can you tell me what's detected here?",
        "assistant_template": "{description}"
    },
    {
        "user": "What kind of waste is present?",
        "assistant_template": "{waste_types}"
    },
    {
        "user": "How many garbage items are there?",
        "assistant_template": "{count_response}"
    },
    {
        "user": "Is there any {class_name} in the image?",
        "assistant_template": "{class_check}"
    },
    {
        "user": "What's the environment like?",
        "assistant_template": "{scene_response}"
    },
    {
        "user": "Where is the garbage located?",
        "assistant_template": "{location_response}"
    },
    {
        "user": "Can you summarize what you found?",
        "assistant_template": "{summary}"
    },
    {
        "user": "Tell me about the scene and the waste.",
        "assistant_template": "{full_analysis}"
    },
]

GARBAGE_DESCRIPTIONS = {
    "plastic": ["plastic waste", "plastic items", "plastic debris", "plastic bottles", "plastic bags", "plastic containers"],
    "glass": ["glass bottles", "glass items", "broken glass", "glass waste", "glass shards", "glass containers"],
    "metal": ["metal cans", "metal waste", "aluminum cans", "tin cans", "metal debris", "metallic items"],
    "paper": ["paper waste", "cardboard", "paper debris", "cardboard boxes", "paper scraps", "discarded paper"],
    "organic": ["organic waste", "food scraps", "biodegradable waste", "organic matter", "food waste", "decomposing material"],
}

SCENE_DESCRIPTIONS = {
    "grass": ["on grassy ground", "in a grassy area", "on the lawn", "surrounded by grass", "on green grass"],
    "sandy": ["on sandy ground", "on the beach", "in sandy terrain", "on sand", "in a sandy area"],
    "rocky": ["on rocky terrain", "among rocks", "on rocky ground", "between stones", "on a rocky surface"],
    "marshy": ["in a marshy area", "in wetlands", "in swampy terrain", "near marsh", "in a wet environment"],
}


def load_cv_models(args):
    print("Loading CV models...")
    
    detector = EnsembleDetector(
        yolo_model_path=args.yolo,
        detr_model_path=args.detr,
        detr_processor_path=args.detr_processor,
        conf_threshold=args.conf_threshold
    )
    
    scene_classifier = None
    if args.scene and Path(args.scene).exists():
        if 'yolo' in args.scene.lower():
            from train_scene_yolo import SceneClassifierYOLO
            scene_classifier = SceneClassifierYOLO(args.scene)
        print(f"Scene classifier loaded: {args.scene}")
    
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


def generate_responses(cv_output):
    detections = cv_output["detections"]
    scene = cv_output["scene"]
    
    counts = {}
    for det in detections:
        label = det["label"]
        counts[label] = counts.get(label, 0) + 1
    
    total = sum(counts.values())
    
    if total == 0:
        description = "I don't see any garbage in this image. The area appears to be clean."
        waste_types = "No waste is detected in the image."
        count_response = "I don't detect any garbage items in this image."
        summary = "The image shows a clean area with no visible garbage."
    else:
        items = []
        for cls, count in counts.items():
            phrase = random.choice(GARBAGE_DESCRIPTIONS.get(cls, [cls]))
            if count == 1:
                items.append(f"{count} piece of {phrase}")
            else:
                items.append(f"{count} pieces of {phrase}")
        
        if len(items) == 1:
            items_str = items[0]
        elif len(items) == 2:
            items_str = f"{items[0]} and {items[1]}"
        else:
            items_str = ", ".join(items[:-1]) + f", and {items[-1]}"
        
        scene_str = ""
        if scene["class"] != "unknown" and scene["confidence"] >= 0.7:
            scene_str = " " + random.choice(SCENE_DESCRIPTIONS.get(scene["class"], [""]))
        
        description_templates = [
            f"I can see {items_str}{scene_str}.",
            f"The image shows {items_str}{scene_str}.",
            f"There {'is' if total == 1 else 'are'} {items_str} detected{scene_str}.",
            f"I've identified {items_str}{scene_str}.",
        ]
        description = random.choice(description_templates)
        
        waste_list = [f"{cls} ({count})" for cls, count in counts.items()]
        waste_types = f"The detected waste types are: {', '.join(waste_list)}."
        
        count_response = f"I count {total} garbage {'item' if total == 1 else 'items'} in total: {', '.join(waste_list)}."
        
        summary = f"Summary: {total} garbage {'item' if total == 1 else 'items'} detected - {items_str}."
    
    if scene["class"] != "unknown" and scene["confidence"] >= 0.7:
        scene_response = f"The environment appears to be {scene['class']} terrain (confidence: {scene['confidence']:.0%})."
        location_response = f"The garbage is located {random.choice(SCENE_DESCRIPTIONS.get(scene['class'], ['in this area']))}."
    else:
        scene_response = "I cannot confidently determine the environment type from this image."
        location_response = "The exact location is unclear from the image."
    
    full_analysis = f"{description} {scene_response}"
    
    return {
        "description": description,
        "waste_types": waste_types,
        "count_response": count_response,
        "scene_response": scene_response,
        "location_response": location_response,
        "summary": summary,
        "full_analysis": full_analysis,
        "counts": counts,
    }


def generate_class_check(counts, class_name):
    actual_classes = list(counts.keys())
    
    if class_name in counts:
        count = counts[class_name]
        return f"Yes, I detected {count} {class_name} {'item' if count == 1 else 'items'} in the image."
    else:
        if counts:
            return f"No, I don't see any {class_name}. I detected: {', '.join(actual_classes)}."
        else:
            return f"No, I don't see any {class_name} or any other garbage in this image."


def create_conversation(cv_output, responses, tokenizer):
    conversations = []
    
    cv_context = f"[CV Output: {json.dumps(cv_output)}]"
    
    template = random.choice(CONVERSATION_TEMPLATES)
    user_msg = template["user"]
    
    if "{class_name}" in user_msg:
        all_classes = list(GARBAGE_DESCRIPTIONS.keys())
        class_name = random.choice(all_classes)
        user_msg = user_msg.format(class_name=class_name)
        assistant_msg = generate_class_check(responses["counts"], class_name)
    else:
        assistant_msg = template["assistant_template"].format(**responses)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{cv_context}\n\n{user_msg}"},
        {"role": "assistant", "content": assistant_msg}
    ]
    
    if random.random() < 0.4 and responses["counts"]:
        followup_templates = [
            ("Can you tell me more about the scene?", responses["scene_response"]),
            ("How confident are you about these detections?", 
             f"The detections have varying confidence levels. {responses['count_response']}"),
            ("Is this area polluted?", 
             f"{'Yes, there is visible pollution with ' + str(sum(responses['counts'].values())) + ' garbage items detected.' if responses['counts'] else 'The area appears clean.'}"),
            ("What should be done about this garbage?",
             f"This garbage should be collected and properly recycled. The {', '.join(responses['counts'].keys())} can typically be recycled at appropriate facilities."),
        ]
        followup = random.choice(followup_templates)
        messages.append({"role": "user", "content": followup[0]})
        messages.append({"role": "assistant", "content": followup[1]})
    
    return tokenizer.apply_chat_template(messages, tokenize=False)


def create_training_dataset(image_dirs, detector, scene_classifier, tokenizer, max_images, variations):
    print("\n" + "="*60)
    print("Creating conversational training dataset...")
    print("="*60)
    
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
    for img_path in tqdm(image_paths, desc="Generating conversations"):
        try:
            cv_output = get_cv_output(str(img_path), detector, scene_classifier)
            responses = generate_responses(cv_output)
            
            for _ in range(variations):
                text = create_conversation(cv_output, responses, tokenizer)
                if text and len(text.strip()) > 0:
                    dataset.append({"text": text})
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Created {len(dataset)} training examples")
    return dataset


def train_model(args):
    print("\n" + "="*60)
    print(f"Training Chat LLM: {args.model}")
    print("="*60)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    detector, scene_classifier = load_cv_models(args)
    
    dataset = create_training_dataset(
        args.train_dirs, detector, scene_classifier, tokenizer, 
        args.max_images, args.variations
    )
    
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.95)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")
    
    print("\n--- Sample conversation ---")
    print(train_data[0]["text"][:1000])
    print("...")
    
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)
    
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        dataloader_num_workers=0,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        max_seq_length=2048,
        dataset_text_field="text",
        packing=False,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nSaving model to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    config = {
        "base_model": args.model,
        "model_type": "causal_lm",
        "lora": True,
        "system_prompt": SYSTEM_PROMPT,
        "max_length": 2048,
        "trained_on": str(datetime.now()),
    }
    with open(f"{args.output}/chat_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Training complete!")
    return model, tokenizer


def test_model(args):
    print("\n" + "="*60)
    print("Testing trained model...")
    print("="*60)
    
    from peft import PeftModel
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.output)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, args.output)
    model.eval()
    
    detector, scene_classifier = load_cv_models(args)
    
    test_images = list(Path(args.test_dir).glob("*.jpg"))[:3]
    
    for img_path in test_images:
        print(f"\n{'='*40}")
        print(f"Image: {img_path.name}")
        
        cv_output = get_cv_output(str(img_path), detector, scene_classifier)
        print(f"CV Output: {json.dumps(cv_output, indent=2)}")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[CV Output: {json.dumps(cv_output)}]\n\nWhat do you see in this image?"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\nAssistant: {response}")


def main():
    parser = argparse.ArgumentParser(description='Train Chat LLM for Garbage Detection')
    
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-14B-Instruct",
                       help="Base model (Qwen2.5-14B-Instruct, mistralai/Mistral-Nemo-Instruct-2407)")
    parser.add_argument('--output', type=str, default="models/chat_llm")
    
    parser.add_argument('--train_dirs', type=str, nargs='+', 
                       default=["data/1206-data/train", "data/1206-data/valid"])
    parser.add_argument('--test_dir', type=str, default="data/1206-data/test")
    
    parser.add_argument('--yolo', type=str, default="models/yolo/yolov8x/best.pt")
    parser.add_argument('--detr', type=str, default="models/rt-detr/rt-detr-101/m")
    parser.add_argument('--detr_processor', type=str, default="models/rt-detr/rt-detr-101/p")
    parser.add_argument('--scene', type=str, default="models/scene_classifier_yolo.pt")
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    
    parser.add_argument('--max_images', type=int, default=2000)
    parser.add_argument('--variations', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    
    parser.add_argument('--test_only', action='store_true')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_model(args)
    else:
        train_model(args)
        test_model(args)


if __name__ == "__main__":
    main()

