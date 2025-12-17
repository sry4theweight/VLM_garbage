"""
Скачивание весов YOLOv8 classification моделей.
Использует ultralytics напрямую - он сам знает откуда качать.
"""
import os
import shutil
from pathlib import Path


def get_ultralytics_cache_dir():
    """Найти директорию кэша ultralytics."""
    # Стандартные места
    possible = [
        Path.home() / ".cache" / "ultralytics",
        Path.home() / "AppData" / "Roaming" / "Ultralytics",  # Windows
        Path(os.environ.get("ULTRALYTICS_DIR", "")) if os.environ.get("ULTRALYTICS_DIR") else None,
    ]
    for p in possible:
        if p and p.exists():
            return p
    return None


def find_weights_in_cache(model_name):
    """Ищем скачанные веса в кэше ultralytics."""
    cache_dir = get_ultralytics_cache_dir()
    if not cache_dir:
        return None
    
    # Ищем файл рекурсивно
    for pt_file in cache_dir.rglob(f"*{model_name}*"):
        if pt_file.suffix == ".pt" and pt_file.stat().st_size > 10_000_000:  # > 10MB
            return pt_file
    return None


def download_yolo_cls_weights(model_name="yolov8x-cls", target_dir="."):
    """
    Скачать веса YOLO classification модели.
    
    Args:
        model_name: имя модели (yolov8n-cls, yolov8s-cls, yolov8m-cls, yolov8l-cls, yolov8x-cls)
        target_dir: куда сохранить файл весов
    
    Returns:
        путь к файлу весов или None
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Имя файла
    if not model_name.endswith(".pt"):
        weights_name = f"{model_name}.pt"
    else:
        weights_name = model_name
        model_name = model_name.replace(".pt", "")
    
    target_path = target_dir / weights_name
    
    # Если уже есть и достаточно большой - не качаем
    if target_path.exists() and target_path.stat().st_size > 50_000_000:  # > 50MB для x
        print(f"✅ Веса уже есть: {target_path} ({target_path.stat().st_size / 1e6:.1f} MB)")
        return target_path
    
    print(f"📦 Загрузка {model_name} через ultralytics...")
    
    try:
        from ultralytics import YOLO
        
        # YOLO автоматически скачает веса при создании модели
        # Передаём просто имя модели без .pt - ultralytics сам разберётся
        model = YOLO(model_name)
        
        # Получаем путь к весам из модели
        weights_path = Path(model.ckpt_path) if hasattr(model, 'ckpt_path') and model.ckpt_path else None
        
        if weights_path and weights_path.exists():
            print(f"✅ Веса загружены: {weights_path}")
            
            # Копируем в целевую директорию если это не то же место
            if weights_path.resolve() != target_path.resolve():
                shutil.copy2(weights_path, target_path)
                print(f"📁 Скопировано в: {target_path}")
            
            return target_path
        
        # Если ckpt_path не сработал, ищем в кэше
        cached = find_weights_in_cache(model_name)
        if cached:
            print(f"✅ Найдено в кэше: {cached}")
            if cached.resolve() != target_path.resolve():
                shutil.copy2(cached, target_path)
                print(f"📁 Скопировано в: {target_path}")
            return target_path
        
        # Последняя попытка - модель могла загрузиться в память, сохраним
        model.save(str(target_path))
        if target_path.exists() and target_path.stat().st_size > 10_000_000:
            print(f"✅ Сохранено: {target_path}")
            return target_path
            
    except Exception as e:
        print(f"❌ Ошибка загрузки через ultralytics: {e}")
    
    return None


def main():
    """Попробовать скачать веса, начиная с самой большой модели."""
    # Порядок попыток: от большой к маленькой
    models_to_try = ["yolov8x-cls", "yolov8l-cls", "yolov8m-cls", "yolov8s-cls"]
    
    for model_name in models_to_try:
        print(f"\n{'='*50}")
        print(f"Попытка: {model_name}")
        print('='*50)
        
        result = download_yolo_cls_weights(model_name, target_dir=".")
        
        if result and result.exists():
            size_mb = result.stat().st_size / 1e6
            print(f"\n✅ УСПЕХ! Скачано: {result} ({size_mb:.1f} MB)")
            print(f"\nДля обучения используйте:")
            print(f"  python train_scene_yolo.py --model {model_name} --batch 64 --epochs 50")
            return result
        
        print(f"⚠️ {model_name} не удалось скачать, пробуем следующую...")
    
    print("\n❌ Не удалось скачать ни одну модель.")
    print("Возможные решения:")
    print("1. Проверьте интернет-соединение")
    print("2. Попробуйте с VPN")
    print("3. Скачайте вручную с https://github.com/ultralytics/assets/releases")
    return None


if __name__ == "__main__":
    main()
