"""
Полный pipeline обучения VLM:
1. Конвертация аннотаций в формат LLaVA
2. Разделение на train/val (опционально)
3. Обучение модели
"""

import argparse
import json
from pathlib import Path

from vlm_training.convert_to_llava_format import convert_to_llava_format
from vlm_training.split_data import split_data


def run_pipeline(
    annotation_file: str,
    output_dir: str,
    base_model: str = "llava-hf/llava-1.5-7b-hf",
    train_ratio: float = 0.9,
    batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    skip_training: bool = False
):
    """
    Полный pipeline подготовки и обучения VLM
    
    Args:
        annotation_file: путь к файлу аннотаций (от annotate_roboflow_dataset.py)
        output_dir: директория для сохранения результатов
        base_model: базовая модель LLaVA
        train_ratio: доля данных для обучения
        batch_size: размер батча
        num_epochs: количество эпох
        learning_rate: скорость обучения
        skip_training: пропустить обучение (только подготовка данных)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PIPELINE ОБУЧЕНИЯ VLM")
    print("=" * 60)
    
    # Шаг 1: Конвертация в формат LLaVA
    print("\n📝 Шаг 1: Конвертация аннотаций в формат LLaVA...")
    llava_file = output_path / "llava_data.json"
    convert_to_llava_format(annotation_file, str(llava_file))
    
    # Шаг 2: Разделение на train/val
    print("\n📊 Шаг 2: Разделение данных на train/val...")
    train_file = output_path / "train.json"
    val_file = output_path / "val.json"
    
    split_data(
        input_file=str(llava_file),
        train_file=str(train_file),
        val_file=str(val_file),
        train_ratio=train_ratio,
        val_ratio=1 - train_ratio
    )
    
    # Проверка данных
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"\n📈 Статистика данных:")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Val samples: {len(val_data)}")
    
    if skip_training:
        print("\n⏭️  Обучение пропущено (--skip_training)")
        print(f"\n✅ Данные подготовлены в: {output_path}")
        print(f"   - LLaVA формат: {llava_file}")
        print(f"   - Train: {train_file}")
        print(f"   - Val: {val_file}")
        return
    
    # Шаг 3: Обучение
    print("\n🚀 Шаг 3: Запуск обучения...")
    print(f"   Базовая модель: {base_model}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    
    # Импортируем здесь, чтобы не загружать torch если только подготовка данных
    from vlm_training.train_vlm import train_vlm
    
    model_output_dir = output_path / "model"
    
    train_vlm(
        train_data_file=str(train_file),
        val_data_file=str(val_file),
        output_dir=str(model_output_dir),
        base_model=base_model,
        use_lora=True,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    print("\n" + "=" * 60)
    print("✅ Pipeline завершён!")
    print(f"   Модель сохранена в: {model_output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Pipeline обучения VLM")
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="data/vlm_annotations/annotations.json",
        help="Путь к файлу аннотаций"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/vlm_training",
        help="Директория для результатов"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Базовая модель LLaVA"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Доля данных для обучения (по умолчанию 0.9)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Размер батча (по умолчанию 2 для экономии памяти)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Количество эпох"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Скорость обучения"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Только подготовка данных, без обучения"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        annotation_file=args.annotation_file,
        output_dir=args.output_dir,
        base_model=args.base_model,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        skip_training=args.skip_training
    )


if __name__ == "__main__":
    main()

