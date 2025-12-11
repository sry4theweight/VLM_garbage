"""
Утилита для разделения данных на train/val/test
"""

import json
import argparse
import random
from pathlib import Path


def split_data(
    input_file: str,
    train_file: str,
    val_file: str,
    test_file: str = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Разделение данных на train/val/test
    
    Args:
        input_file: входной JSON файл с разметкой
        train_file: путь для сохранения train данных
        val_file: путь для сохранения val данных
        test_file: путь для сохранения test данных (опционально)
        train_ratio: доля train данных (по умолчанию 0.8)
        val_ratio: доля val данных (по умолчанию 0.1)
        seed: seed для random
    """
    random.seed(seed)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    random.shuffle(data)
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    
    if test_file:
        test_data = data[train_size + val_size:]
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"Test: {len(test_data)} samples")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Total: {total} samples")
    print(f"Train: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
    print(f"Val: {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
    if test_file:
        print(f"Test: {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Разделение данных на train/val/test")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Входной JSON файл с разметкой"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Путь для сохранения train данных"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        required=True,
        help="Путь для сохранения val данных"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Путь для сохранения test данных (опционально)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Доля train данных (по умолчанию 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Доля val данных (по умолчанию 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для random"
    )
    
    args = parser.parse_args()
    
    split_data(
        args.input_file,
        args.train_file,
        args.val_file,
        args.test_file,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )


if __name__ == "__main__":
    main()

