"""
Конвертер разметки в формат LLaVA для обучения VLM

Каждая пара вопрос-ответ становится отдельным примером обучения.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Обрезать текст до максимальной длины"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."


def convert_to_llava_format(annotation_file: str, output_file: str, max_qa_per_image: int = 3):
    """
    Конвертация JSON разметки в формат LLaVA
    
    Каждая пара Q&A становится отдельным примером обучения.
    
    Формат LLaVA:
    {
        "id": "unique_id",
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\n{question}"},
            {"from": "gpt", "value": "{answer}"}
        ]
    }
    
    Args:
        annotation_file: путь к файлу аннотаций
        output_file: путь для сохранения
        max_qa_per_image: максимум QA пар на изображение (чтобы не было слишком много)
    """
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    llava_data = []
    sample_id = 0
    
    for ann in annotations:
        image_path = ann['image']
        detections = ann.get('detections', [])
        qa_pairs = ann.get('qa_pairs', [])
        
        # 1. Общее описание объектов (если есть детекции)
        if detections:
            descriptions = [det['description'] for det in detections if det.get('description')]
            if descriptions:
                answer = truncate_text(" ".join(descriptions), max_chars=400)
                llava_data.append({
                    "id": f"sample_{sample_id}",
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": "<image>\nDescribe the garbage objects in this image."},
                        {"from": "gpt", "value": answer}
                    ]
                })
                sample_id += 1
        
        # 2. Вопрос о количестве объектов
        count_qa = [qa for qa in qa_pairs if "How many objects" in qa.get('question', '')]
        if count_qa:
            qa = count_qa[0]
            llava_data.append({
                "id": f"sample_{sample_id}",
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{qa['question']}"},
                    {"from": "gpt", "value": qa['answer']}
                ]
            })
            sample_id += 1
        
        # 3. Вопросы о наличии конкретных классов (ограничиваем количество)
        presence_qa = [qa for qa in qa_pairs if "Is there any" in qa.get('question', '')]
        for qa in presence_qa[:max_qa_per_image]:
            llava_data.append({
                "id": f"sample_{sample_id}",
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{qa['question']}"},
                    {"from": "gpt", "value": qa['answer']}
                ]
            })
            sample_id += 1
        
        # 4. Описание отдельных объектов с bbox (ограничиваем)
        for det in detections[:2]:  # Максимум 2 детекции на изображение
            bbox = det.get('bbox', [0, 0, 0, 0])
            label = det.get('label', 'object')
            description = det.get('description', '')
            
            if description:
                bbox_str = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
                answer = truncate_text(description, max_chars=300)
                llava_data.append({
                    "id": f"sample_{sample_id}",
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\nThere is a {label} at {bbox_str}. Describe it."},
                        {"from": "gpt", "value": answer}
                    ]
                })
                sample_id += 1
    
    # Сохранение
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(llava_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(llava_data)} training samples from {len(annotations)} images")
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Конвертация разметки в формат LLaVA")
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Путь к JSON файлу с разметкой"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Путь к выходному JSON файлу в формате LLaVA"
    )
    
    args = parser.parse_args()
    convert_to_llava_format(args.annotation_file, args.output_file)


if __name__ == "__main__":
    main()

