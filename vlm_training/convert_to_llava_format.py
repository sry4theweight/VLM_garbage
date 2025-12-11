"""
Конвертер разметки в формат LLaVA для обучения VLM
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_to_llava_format(annotation_file: str, output_file: str):
    """
    Конвертация JSON разметки в формат LLaVA
    
    Формат LLaVA:
    {
        "id": "unique_id",
        "image": "path/to/image.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n{question}"
            },
            {
                "from": "gpt",
                "value": "{answer}"
            }
        ]
    }
    """
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    llava_data = []
    
    for idx, ann in enumerate(annotations):
        image_path = ann['image']
        detections = ann.get('detections', [])
        qa_pairs = ann.get('qa_pairs', [])
        
        # Создаем conversations на основе описаний объектов
        conversations = []
        
        # Добавляем описания объектов как диалог
        if detections:
            description_text = "Describe the objects in this image:\n"
            answer_text = ""
            
            for det in detections:
                description_text += f"- There is a {det['label']} at coordinates {det['bbox']}. "
                answer_text += f"{det['description']} "
            
            conversations.append({
                "from": "human",
                "value": f"<image>\n{description_text.strip()}"
            })
            conversations.append({
                "from": "gpt",
                "value": answer_text.strip()
            })
        
        # Добавляем Q&A пары
        for qa in qa_pairs:
            conversations.append({
                "from": "human",
                "value": f"<image>\n{qa['question']}"
            })
            conversations.append({
                "from": "gpt",
                "value": qa['answer']
            })
        
        # Если есть описания объектов с bbox
        for det in detections:
            bbox_str = f"[{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]"
            conversations.append({
                "from": "human",
                "value": f"<image>\nThere is a {det['label']} at coordinates {bbox_str}. Describe where it is and what is around it."
            })
            conversations.append({
                "from": "gpt",
                "value": det['description']
            })
        
        if conversations:
            llava_data.append({
                "id": f"annotation_{idx}",
                "image": image_path,
                "conversations": conversations
            })
    
    # Сохранение
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(llava_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(llava_data)} annotations to LLaVA format")
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

