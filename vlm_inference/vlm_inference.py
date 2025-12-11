"""
Скрипт для использования обученной VLM модели для ответов на вопросы по изображениям
"""

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


class VLMInference:
    """Инференс для VLM модели"""
    
    def __init__(self, model_path: str, base_model: str = None, device: str = None):
        """
        Args:
            model_path: путь к обученной модели (или путь к LoRA весам)
            base_model: базовая модель (если использовался LoRA)
            device: устройство для вычислений
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from {model_path}...")
        
        if base_model:
            # Загрузка базовой модели и LoRA весов
            base = LlavaForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model = self.model.merge_and_unload()
            self.processor = AutoProcessor.from_pretrained(base_model)
        else:
            # Загрузка полной модели
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def answer_question(self, image_path: str, question: str) -> str:
        """
        Ответ на вопрос по изображению
        
        Args:
            image_path: путь к изображению
            question: вопрос
        
        Returns:
            Ответ модели
        """
        image = Image.open(image_path).convert('RGB')
        
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
    
    def describe_object(self, image_path: str, bbox: list, label: str) -> str:
        """
        Описание объекта с учетом координат bbox
        
        Args:
            image_path: путь к изображению
            bbox: координаты [x1, y1, x2, y2]
            label: метка объекта (например, 'glass')
        
        Returns:
            Описание объекта и его окружения
        """
        question = f"There is a {label} at coordinates [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]. Describe where it is located and what is around it."
        return self.answer_question(image_path, question)


def main():
    parser = argparse.ArgumentParser(description="Инференс VLM модели")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Путь к обученной модели"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Базовая модель (если использовался LoRA)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Путь к изображению"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Вопрос для модели"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        help="Координаты bbox [x1, y1, x2, y2] (опционально)"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Метка объекта (используется с --bbox)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Создание инференса
    vlm = VLMInference(
        model_path=args.model_path,
        base_model=args.base_model,
        device=args.device
    )
    
    # Генерация ответа
    if args.bbox and args.label:
        answer = vlm.describe_object(args.image, args.bbox, args.label)
        print(f"\nDescription: {answer}")
    elif args.question:
        answer = vlm.answer_question(args.image, args.question)
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {answer}")
    else:
        print("Please provide either --question or --bbox with --label")


if __name__ == "__main__":
    main()

