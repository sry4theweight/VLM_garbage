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
from peft import LoraConfig, get_peft_model
import os


def check_gpu():
    """Проверка доступности и информации о GPU"""
    print("=" * 60)
    print("ПРОВЕРКА GPU")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA доступна!")
        print(f"   Количество GPU: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            gpu_memory_free = gpu_memory - gpu_memory_allocated
            
            print(f"\n   GPU {i}: {gpu_name}")
            print(f"   - Общая память: {gpu_memory:.2f} GB")
            print(f"   - Свободная память: {gpu_memory_free:.2f} GB")
            print(f"   - Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Текущий GPU
        current_device = torch.cuda.current_device()
        print(f"\n   Текущий GPU: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Версии
        print(f"\n   PyTorch версия: {torch.__version__}")
        print(f"   CUDA версия: {torch.version.cuda}")
        if hasattr(torch.backends, 'cudnn'):
            print(f"   cuDNN версия: {torch.backends.cudnn.version()}")
            print(f"   cuDNN включен: {torch.backends.cudnn.enabled}")
        
        print("=" * 60)
        return True
    else:
        print("❌ CUDA НЕ доступна!")
        print("   Обучение будет происходить на CPU (очень медленно)")
        print(f"\n   PyTorch версия: {torch.__version__}")
        
        # Проверка причины
        if not torch.cuda.is_available():
            print("\n   Возможные причины:")
            print("   1. Не установлены NVIDIA драйверы")
            print("   2. PyTorch установлен без поддержки CUDA")
            print("   3. Нет совместимой NVIDIA GPU")
            print("\n   Для установки PyTorch с CUDA через Conda:")
            print("   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
        
        print("=" * 60)
        return False


class LLaVADataset(Dataset):
    """Датасет для обучения LLaVA"""
    
    def __init__(self, data_file: str, processor, image_dir: str = None, max_length: int = 2048):
        """
        Args:
            data_file: путь к JSON файлу с данными в формате LLaVA
            processor: процессор модели
            image_dir: базовый путь к директории с изображениями (если пути в JSON относительные)
            max_length: максимальная длина последовательности
        """
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.processor = processor
        self.image_dir = Path(image_dir) if image_dir else None
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        
        # Обработка пути к изображению (Windows и Unix)
        image_path = image_path.replace('\\', '/')
        
        if self.image_dir and not Path(image_path).is_absolute():
            image_path = self.image_dir / image_path
        else:
            image_path = Path(image_path)
        
        # Загрузка изображения
        image = Image.open(image_path).convert('RGB')
        
        # Формирование текста в формате LLaVA
        # Формат: "USER: <image>\n{question}\nASSISTANT: {answer}"
        conversations = item['conversations']
        
        # Предполагаем структуру: [human, gpt] пары
        prompt_parts = []
        for i, conv in enumerate(conversations):
            if conv['from'] == 'human':
                prompt_parts.append(f"USER: {conv['value']}")
            else:
                prompt_parts.append(f"ASSISTANT: {conv['value']}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Обработка с процессором (без truncation чтобы не потерять <image> токен)
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length
        )
        
        # Удаляем batch dimension для Dataset
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Ручная обрезка если нужно (сохраняя первые токены с <image>)
        if inputs['input_ids'].shape[0] > self.max_length:
            inputs['input_ids'] = inputs['input_ids'][:self.max_length]
            inputs['attention_mask'] = inputs['attention_mask'][:self.max_length]
        
        # Создаём labels (копия input_ids, но -100 для padding)
        labels = inputs['input_ids'].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        
        return inputs


def collate_fn(batch):
    """Функция для батчинга"""
    # Все примеры уже имеют одинаковую длину благодаря padding в __getitem__
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
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
    # Проверка GPU перед обучением
    has_gpu = check_gpu()
    
    if not has_gpu:
        user_input = input("\n⚠️  GPU не обнаружен. Продолжить обучение на CPU? (y/n): ")
        if user_input.lower() != 'y':
            print("Обучение отменено.")
            return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nИспользуемое устройство: {device}")
    
    # Загрузка модели и процессора
    print(f"Loading model: {base_model}...")
    processor = AutoProcessor.from_pretrained(base_model, use_fast=True)
    
    # Загружаем модель на одно устройство (без device_map для корректного backward)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    
    if device == "cuda":
        model = model.to(device)
    
    # Применение LoRA если нужно
    if use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
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
        eval_strategy="steps" if val_dataset else "no",
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

