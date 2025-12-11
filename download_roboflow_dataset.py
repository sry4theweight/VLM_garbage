"""
Скрипт для скачивания датасета Roboflow для обучения VLM
"""

import argparse
import os
from pathlib import Path
from roboflow import Roboflow


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: str,
    format: str = "coco"
):
    """
    Скачивание датасета Roboflow
    
    Args:
        api_key: API ключ Roboflow (можно оставить пустым и использовать переменную окружения ROBOFLOW_API_KEY)
        workspace: имя workspace в Roboflow
        project: имя проекта в Roboflow
        version: версия датасета
        output_dir: директория для сохранения датасета
        format: формат датасета ('coco' или 'yolov8')
    """
    # Если API key не указан, пытаемся получить из переменной окружения
    if not api_key:
        api_key = os.getenv("ROBOFLOW_API_KEY", "")
        if not api_key:
            print("Warning: Roboflow API key not provided. Trying to download without key...")
    
    # Создаем директорию если не существует
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Roboflow dataset:")
    print(f"  Workspace: {workspace}")
    print(f"  Project: {project}")
    print(f"  Version: {version}")
    print(f"  Format: {format}")
    print(f"  Output directory: {output_dir}")
    
    # Инициализация Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Получение проекта и версии
    roboflow_project = rf.workspace(workspace).project(project)
    roboflow_version = roboflow_project.version(version)
    
    # Скачивание датасета
    dataset = roboflow_version.download(format, location=str(output_path))
    
    print(f"\nDataset downloaded successfully!")
    print(f"Dataset location: {dataset.location}")
    print(f"\nDataset structure:")
    print(f"  - Train: {dataset.location}/train/")
    print(f"  - Valid: {dataset.location}/valid/")
    print(f"  - Test: {dataset.location}/test/")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Скачивание датасета Roboflow для обучения VLM")
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API ключ Roboflow (можно не указывать, будет использована переменная окружения ROBOFLOW_API_KEY)"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="kuivashev",
        help="Имя workspace в Roboflow"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="my-normal-dataset",
        help="Имя проекта в Roboflow"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Версия датасета"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/roboflow_dataset",
        help="Директория для сохранения датасета"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="coco",
        choices=["coco", "yolov8"],
        help="Формат датасета"
    )
    
    args = parser.parse_args()
    
    download_roboflow_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        output_dir=args.output_dir,
        format=args.format
    )


if __name__ == "__main__":
    main()

