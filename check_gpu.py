"""
Скрипт для проверки доступности GPU и системной конфигурации
"""

import torch
import sys
import os


def get_conda_info():
    """Получение информации о Conda окружении"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    conda_prefix = os.environ.get('CONDA_PREFIX', None)
    return conda_env, conda_prefix


def check_gpu():
    """Полная проверка GPU и системной конфигурации"""
    print("=" * 60)
    print("ПРОВЕРКА GPU И СИСТЕМНОЙ КОНФИГУРАЦИИ")
    print("=" * 60)
    
    # Conda окружение
    conda_env, conda_prefix = get_conda_info()
    if conda_env:
        print(f"\n🐍 Conda окружение: {conda_env}")
        print(f"   Путь: {conda_prefix}")
    else:
        print("\n🐍 Conda окружение: не обнаружено")
    
    # Python версия
    print(f"\n📌 Python версия: {sys.version}")
    print(f"📌 PyTorch версия: {torch.__version__}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n✅ CUDA доступна!")
        print(f"   CUDA версия: {torch.version.cuda}")
        print(f"   Количество GPU: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_name = props.name
            gpu_memory = props.total_memory / (1024**3)  # GB
            
            print(f"\n   🖥️  GPU {i}: {gpu_name}")
            print(f"   - Общая память: {gpu_memory:.2f} GB")
            print(f"   - Compute Capability: {props.major}.{props.minor}")
            print(f"   - Мультипроцессоров: {props.multi_processor_count}")
        
        # cuDNN
        if hasattr(torch.backends, 'cudnn'):
            print(f"\n   cuDNN версия: {torch.backends.cudnn.version()}")
            print(f"   cuDNN включен: {torch.backends.cudnn.enabled}")
        
        # Текущее использование памяти
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
        reserved = torch.cuda.memory_reserved(current_device) / (1024**3)
        print(f"\n   Текущее использование памяти GPU {current_device}:")
        print(f"   - Выделено: {allocated:.2f} GB")
        print(f"   - Зарезервировано: {reserved:.2f} GB")
        
        # Тест работы GPU
        print("\n   🧪 Тест работы GPU...")
        try:
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print("   ✅ Тест пройден успешно!")
        except Exception as e:
            print(f"   ❌ Ошибка теста: {e}")
        
        print("\n" + "=" * 60)
        print("✅ GPU готов к обучению!")
        print("=" * 60)
        return True
    else:
        print("\n❌ CUDA НЕ доступна!")
        print("   Обучение будет происходить на CPU (очень медленно)")
        
        print("\n   Возможные причины:")
        print("   1. Не установлены NVIDIA драйверы")
        print("   2. PyTorch установлен без поддержки CUDA")
        print("   3. Нет совместимой NVIDIA GPU")
        
        print("\n   📋 Для установки PyTorch с CUDA через Conda:")
        print("   # CUDA 11.8:")
        print("   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("\n   # CUDA 12.1:")
        print("   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia")
        print("\n   # Или через pip в Conda окружении:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n" + "=" * 60)
        print("❌ GPU не доступен для обучения")
        print("=" * 60)
        return False


def check_memory_for_model(model_size_gb: float = 14.0):
    """
    Проверка достаточности памяти GPU для модели
    
    Args:
        model_size_gb: примерный размер модели в GB (по умолчанию 14GB для LLaVA-7B)
    """
    if not torch.cuda.is_available():
        print("GPU не доступен")
        return False
    
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / (1024**3)
    
    print(f"\n📊 Проверка памяти для модели:")
    print(f"   Доступная память GPU: {total_memory:.2f} GB")
    print(f"   Требуемая память (примерно): {model_size_gb:.2f} GB")
    
    if total_memory >= model_size_gb:
        print(f"   ✅ Памяти достаточно!")
        return True
    else:
        print(f"   ⚠️  Памяти может не хватить!")
        print(f"   Рекомендации:")
        print(f"   - Используйте LoRA для уменьшения требований к памяти")
        print(f"   - Уменьшите batch_size")
        print(f"   - Используйте gradient checkpointing")
        return False


if __name__ == "__main__":
    has_gpu = check_gpu()
    
    if has_gpu:
        # Проверка памяти для LLaVA-7B (~14GB VRAM)
        check_memory_for_model(14.0)

