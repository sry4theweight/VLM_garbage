"""
Обучение классификатора сцен для VLM
Классы: grass, road, sand, water, floor, indoor

Этот классификатор дополняет ваш ансамбль детекции мусора,
позволяя описывать сцену: "пластик на траве"
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

# Классы сцен
SCENE_CLASSES = ['grass', 'road', 'sand', 'water', 'floor', 'indoor', 'outdoor_other']


class SceneDataset(Dataset):
    """Датасет для классификации сцен"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Собираем изображения из папок по классам
        for class_idx, class_name in enumerate(SCENE_CLASSES):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, class_idx))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SceneClassifier(nn.Module):
    """Классификатор сцен на основе MobileNetV3 (лёгкая модель)"""
    
    def __init__(self, num_classes=len(SCENE_CLASSES)):
        super().__init__()
        # Используем MobileNetV3 - быстрая и лёгкая
        self.backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        
        # Заменяем последний слой
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def create_scene_dataset_from_roboflow(output_dir="data/scene_dataset"):
    """
    Создаёт структуру папок для датасета сцен.
    Вам нужно будет вручную добавить изображения в папки.
    
    ИЛИ можно использовать готовый датасет:
    - Places365: http://places2.csail.mit.edu/
    - Scene Recognition: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in SCENE_CLASSES:
        (output_dir / "train" / class_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "val" / class_name).mkdir(parents=True, exist_ok=True)
    
    print(f"Создана структура папок в {output_dir}")
    print("\nДобавьте изображения в папки:")
    for class_name in SCENE_CLASSES:
        print(f"  - {output_dir}/train/{class_name}/")
    print("\nИЛИ скачайте датасет Intel Image Classification:")
    print("  https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
    
    return output_dir


def train_scene_classifier(
    train_dir: str,
    val_dir: str = None,
    output_path: str = "models/scene_classifier.pt",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001
):
    """Обучение классификатора сцен"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Трансформации
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Датасеты
    train_dataset = SceneDataset(train_dir, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    val_loader = None
    if val_dir:
        val_dataset = SceneDataset(val_dir, val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Модель
    model = SceneClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
        
        scheduler.step()
        
        # Validation
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            print(f"Validation Accuracy: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'classes': SCENE_CLASSES,
                    'accuracy': val_acc
                }, output_path)
                print(f"Saved best model with {val_acc:.2f}% accuracy")
    
    # Сохраняем финальную модель если нет валидации
    if not val_loader:
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': SCENE_CLASSES,
        }, output_path)
        print(f"Saved model to {output_path}")
    
    return model


class SceneClassifierInference:
    """Инференс классификатора сцен"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint.get('classes', SCENE_CLASSES)
        
        self.model = SceneClassifier(len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """Предсказание класса сцены"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        
        return {
            'class': self.classes[pred.item()],
            'confidence': conf.item(),
            'all_probs': {cls: probs[0, i].item() for i, cls in enumerate(self.classes)}
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train scene classifier")
    parser.add_argument("--create_structure", action="store_true", help="Create folder structure for dataset")
    parser.add_argument("--train_dir", type=str, default="data/scene_dataset/train", help="Training data directory")
    parser.add_argument("--val_dir", type=str, default="data/scene_dataset/val", help="Validation data directory")
    parser.add_argument("--output", type=str, default="models/scene_classifier.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_scene_dataset_from_roboflow()
    else:
        train_scene_classifier(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

