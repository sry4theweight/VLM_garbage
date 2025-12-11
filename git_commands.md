# Команды Git для работы с проектом

## Обновление проекта в Git (Push)

После внесения изменений в проект, выполните следующие команды для обновления в Git:

### 1. Проверка статуса

```bash
git status
```

### 2. Добавление всех изменений

```bash
# Добавить все изменения
git add .

# Или добавить конкретные файлы
git add vlm_annotation/
git add vlm_training/
git add vlm_inference/
git add download_roboflow_dataset.py
git add annotate_roboflow_dataset.py
git add README.md
git add requirements_vlm.txt
```

### 3. Коммит изменений

```bash
git commit -m "Add VLM annotation and training pipeline

- Add ensemble detector for object detection
- Add data annotation script with LLM integration
- Add VLM training scripts with LoRA support
- Add inference script for VLM
- Add Roboflow dataset download script
- Update README with VLM usage instructions"
```

### 4. Push в удаленный репозиторий

```bash
# Если вы работаете с веткой main/master
git push origin main

# Или если используете другую ветку
git push origin your-branch-name
```

### 5. Если нужно принудительно обновить (осторожно!)

```bash
# Только если вы уверены, что хотите перезаписать удаленную ветку
git push --force origin main
```

---

## Обновление проекта из Git (Pull)

Для обновления локальной копии проекта из Git:

### 1. Проверка текущей ветки

```bash
git branch
```

### 2. Получение изменений из удаленного репозитория

```bash
# Получить информацию об изменениях
git fetch origin

# Посмотреть что изменилось
git log HEAD..origin/main
```

### 3. Обновление локальной копии

```bash
# Если нет локальных изменений - просто pull
git pull origin main

# Или если нужно обновить конкретную ветку
git pull origin your-branch-name
```

### 4. Если есть конфликты

```bash
# Git покажет конфликты, разрешите их вручную в файлах
# После разрешения конфликтов:
git add .
git commit -m "Resolve merge conflicts"
git push origin main
```

### 5. Обновление всех веток

```bash
git fetch --all
git pull --all
```

---

## Быстрые команды (одной строкой)

### Обновить проект в Git (после изменений):
```bash
git add . && git commit -m "Update VLM pipeline" && git push origin main
```

### Обновить проект из Git:
```bash
git pull origin main
```

---

## Полезные команды

### Посмотреть историю коммитов
```bash
git log --oneline --graph
```

### Посмотреть различия перед коммитом
```bash
git diff
```

### Отменить изменения в файле (еще не закоммиченные)
```bash
git checkout -- filename
```

### Создать новую ветку
```bash
git checkout -b feature/vlm-enhancements
```

### Переключиться на ветку
```bash
git checkout main
```

### Удалить ветку
```bash
git branch -d branch-name  # локальная ветка
git push origin --delete branch-name  # удаленная ветка
```

---

## Настройка .gitignore

Убедитесь, что в `.gitignore` есть следующие исключения (чтобы не загружать большие файлы):

```
# Модели и данные
models/*.pt
models/*.safetensors
data/
*.ckpt
*.pth

# Датасеты
roboflow_dataset/
vlm_annotations/
*.json

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
venv/
env/
ENV/
.venv

# Logs
*.log
logs/
wandb/

# OS
.DS_Store
Thumbs.db
```

---

## Первоначальная настройка (если еще не настроено)

### Если проект еще не подключен к Git:

```bash
# Инициализация репозитория
git init

# Добавление удаленного репозитория
git remote add origin https://github.com/your-username/your-repo.git

# Первый коммит
git add .
git commit -m "Initial commit: VLM pipeline for object detection"

# Первый push
git push -u origin main
```

### Если нужно клонировать проект:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

