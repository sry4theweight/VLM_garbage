"""
Скрипт для fine-tuning VLM модели (LLaVA) на размеченных данных
"""

import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import os


class LLaVADataset(Dataset):
    """Датасет для обучения LLaVA"""
    
    def __init__(self, data_file: str, processor, image_dir: str = None):
        """
        Args:
            data_file: путь к JSON файлу с данными в формате LLaVA
            processor: процессор модели
            image_dir: базовый путь к директории с изображениями (если пути в JSON относительные)
        """
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.processor = processor
        self.image_dir = Path(image_dir) if image_dir else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        
        # Обработка пути к изображению
        if self.image_dir and not Path(image_path).is_absolute():
            image_path = self.image_dir / image_path
        else:
            image_path = Path(image_path)
        
        # Загрузка изображения
        image = Image.open(image_path).convert('RGB')
        
        # Формирование текста разговора
        conversations = item['conversations']
        full_text = ""
        for conv in conversations:
            role = "Human" if conv['from'] == 'human' else "Assistant"
            full_text += f"{role}: {conv['value']}\n"
        
        # Обработка с процессором
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Удаляем batch dimension для Dataset
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs


def collate_fn(batch):
    """Функция для батчинга"""
    images = [item['pixel_values'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Паддинг
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = []
    padded_attention_mask = []
    
    for ids, mask in zip(input_ids, attention_mask):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
        padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
    
    return {
        'pixel_values': torch.stack(images),
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask)
    }


def train_vlm(
    train_data_file: str,
    val_data_file: str,
    output_dir: str,
    base_model: str = "llava-hf/llava-1.5-7b-hf",
    image_dir: str = None,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    save_steps: int = 500,
    logging_steps: int = 100
):
    """
    Fine-tuning VLM модели
    
    Args:
        train_data_file: путь к JSON файлу с обучающими данными
        val_data_file: путь к JSON файлу с валидационными данными (может быть None)
        output_dir: директория для сохранения модели
        base_model: базовая модель из HuggingFace
        image_dir: базовый путь к директории с изображениями
        use_lora: использовать LoRA для fine-tuning
        lora_r: ранг LoRA
        lora_alpha: alpha параметр LoRA
        batch_size: размер батча
        learning_rate: скорость обучения
        num_epochs: количество эпох
        save_steps: частота сохранения
        logging_steps: частота логирования
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Загрузка модели и процессора
    print(f"Loading model: {base_model}...")
    processor = AutoProcessor.from_pretrained(base_model)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # Применение LoRA если нужно
    if use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CONDITIONAL_GENERATION,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Создание датасетов
    print("Loading datasets...")
    train_dataset = LLaVADataset(train_data_file, processor, image_dir)
    
    val_dataset = None
    if val_data_file and Path(val_data_file).exists():
        val_dataset = LLaVADataset(val_data_file, processor, image_dir)
    
    # Параметры обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        fp16=device == "cuda",
        logging_steps=logging_steps,
        save_steps=save_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Обучение
    print("Starting training...")
    trainer.train()
    
    # Сохранение
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning VLM модели")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Путь к JSON файлу с обучающими данными (формат LLaVA)"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Путь к JSON файлу с валидационными данными (опционально)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Директория для сохранения обученной модели"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Базовая модель из HuggingFace"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Базовый путь к директории с изображениями (если пути в JSON относительные)"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Использовать LoRA для fine-tuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Ранг LoRA"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha параметр LoRA"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Размер батча"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Скорость обучения"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Количество эпох"
    )
    
    args = parser.parse_args()
    
    train_vlm(
        train_data_file=args.train_data,
        val_data_file=args.val_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        image_dir=args.image_dir,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )


if __name__ == "__main__":
    main()

