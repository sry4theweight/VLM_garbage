"""
GUI для просмотра результатов разметки изображений
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
import random


# Цвета для разных классов
CLASS_COLORS = {
    'glass': '#00FF00',      # Зелёный
    'plastic': '#FF6B6B',    # Красный
    'metal': '#4ECDC4',      # Бирюзовый
    'paper': '#FFE66D',      # Жёлтый
    'organic': '#95E1D3',    # Мятный
}


class AnnotationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("VLM Annotation Viewer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Данные
        self.annotations = []
        self.current_index = 0
        self.show_bboxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_confidence = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        # Стили
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=6, font=('Segoe UI', 10))
        style.configure('TLabel', background='#2b2b2b', foreground='white', font=('Segoe UI', 10))
        style.configure('TCheckbutton', background='#2b2b2b', foreground='white', font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.configure(style='TFrame')
        style.configure('TFrame', background='#2b2b2b')
        
        # Верхняя панель с кнопками
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(top_frame, text="📂 Открыть JSON", command=self.load_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="◀ Назад", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="▶ Вперёд", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="🎲 Случайное", command=self.random_image).pack(side=tk.LEFT, padx=5)
        
        # Счётчик
        self.counter_label = ttk.Label(top_frame, text="0 / 0", style='Header.TLabel')
        self.counter_label.pack(side=tk.LEFT, padx=20)
        
        # Чекбоксы для отображения
        ttk.Checkbutton(top_frame, text="Показать bbox", variable=self.show_bboxes, 
                       command=self.refresh_image).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(top_frame, text="Показать метки", variable=self.show_labels,
                       command=self.refresh_image).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(top_frame, text="Показать confidence", variable=self.show_confidence,
                       command=self.refresh_image).pack(side=tk.RIGHT, padx=5)
        
        # Основная область
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель - изображение
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas для изображения
        self.canvas = tk.Canvas(left_frame, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Правая панель - информация
        right_frame = ttk.Frame(content_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Путь к файлу
        ttk.Label(right_frame, text="📁 Файл:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.file_label = ttk.Label(right_frame, text="-", wraplength=380)
        self.file_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Детекции
        ttk.Label(right_frame, text="🎯 Детекции:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        # Список детекций с прокруткой
        det_frame = ttk.Frame(right_frame)
        det_frame.pack(fill=tk.X, pady=(0, 15))
        
        det_scroll = ttk.Scrollbar(det_frame)
        det_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.detections_text = tk.Text(det_frame, height=10, wrap=tk.WORD, 
                                       bg='#1e1e1e', fg='white', font=('Consolas', 9),
                                       yscrollcommand=det_scroll.set)
        self.detections_text.pack(fill=tk.X)
        det_scroll.config(command=self.detections_text.yview)
        
        # Описания
        ttk.Label(right_frame, text="📝 Описания:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        desc_frame = ttk.Frame(right_frame)
        desc_frame.pack(fill=tk.X, pady=(0, 15))
        
        desc_scroll = ttk.Scrollbar(desc_frame)
        desc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.descriptions_text = tk.Text(desc_frame, height=8, wrap=tk.WORD,
                                         bg='#1e1e1e', fg='#90EE90', font=('Consolas', 9),
                                         yscrollcommand=desc_scroll.set)
        self.descriptions_text.pack(fill=tk.X)
        desc_scroll.config(command=self.descriptions_text.yview)
        
        # Q&A
        ttk.Label(right_frame, text="❓ Вопросы и ответы:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        qa_frame = ttk.Frame(right_frame)
        qa_frame.pack(fill=tk.BOTH, expand=True)
        
        qa_scroll = ttk.Scrollbar(qa_frame)
        qa_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.qa_text = tk.Text(qa_frame, wrap=tk.WORD,
                               bg='#1e1e1e', fg='#87CEEB', font=('Consolas', 9),
                               yscrollcommand=qa_scroll.set)
        self.qa_text.pack(fill=tk.BOTH, expand=True)
        qa_scroll.config(command=self.qa_text.yview)
        
        # Легенда цветов
        legend_frame = ttk.Frame(right_frame)
        legend_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(legend_frame, text="Легенда:", style='Header.TLabel').pack(anchor=tk.W)
        
        legend_inner = ttk.Frame(legend_frame)
        legend_inner.pack(fill=tk.X, pady=5)
        
        for i, (cls, color) in enumerate(CLASS_COLORS.items()):
            frame = ttk.Frame(legend_inner)
            frame.grid(row=i//3, column=i%3, padx=5, pady=2, sticky=tk.W)
            
            color_box = tk.Canvas(frame, width=15, height=15, bg=color, highlightthickness=1)
            color_box.pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Label(frame, text=cls, font=('Segoe UI', 8)).pack(side=tk.LEFT)
    
    def setup_bindings(self):
        """Настройка горячих клавиш"""
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.next_image())
        self.root.bind('r', lambda e: self.random_image())
        self.root.bind('<Control-o>', lambda e: self.load_annotations())
        self.canvas.bind('<Configure>', lambda e: self.refresh_image())
    
    def load_annotations(self):
        """Загрузка файла аннотаций"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл аннотаций",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="data/vlm_annotations"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                self.current_index = 0
                self.show_current_annotation()
                messagebox.showinfo("Успех", f"Загружено {len(self.annotations)} аннотаций")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")
    
    def show_current_annotation(self):
        """Отображение текущей аннотации"""
        if not self.annotations:
            return
        
        annotation = self.annotations[self.current_index]
        
        # Обновляем счётчик
        self.counter_label.config(text=f"{self.current_index + 1} / {len(self.annotations)}")
        
        # Путь к файлу
        image_path = annotation.get('image', '-')
        self.file_label.config(text=Path(image_path).name)
        
        # Загружаем и отображаем изображение
        self.load_and_display_image(image_path, annotation.get('detections', []))
        
        # Детекции
        self.detections_text.delete(1.0, tk.END)
        detections = annotation.get('detections', [])
        for i, det in enumerate(detections):
            label = det.get('label', 'unknown')
            conf = det.get('confidence', 0)
            bbox = det.get('bbox', [0, 0, 0, 0])
            
            self.detections_text.insert(tk.END, f"[{i+1}] {label}\n")
            self.detections_text.insert(tk.END, f"    Confidence: {conf:.2%}\n")
            self.detections_text.insert(tk.END, f"    BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]\n\n")
        
        # Описания
        self.descriptions_text.delete(1.0, tk.END)
        for i, det in enumerate(detections):
            desc = det.get('description', '')
            label = det.get('label', 'unknown')
            self.descriptions_text.insert(tk.END, f"[{label}]: {desc}\n\n")
        
        # Q&A
        self.qa_text.delete(1.0, tk.END)
        qa_pairs = annotation.get('qa_pairs', [])
        for qa in qa_pairs:
            q = qa.get('question', '')
            a = qa.get('answer', '')
            self.qa_text.insert(tk.END, f"Q: {q}\n")
            self.qa_text.insert(tk.END, f"A: {a}\n\n")
    
    def load_and_display_image(self, image_path, detections):
        """Загрузка и отображение изображения с bbox"""
        try:
            # Загрузка изображения
            image = Image.open(image_path).convert('RGB')
            
            # Рисуем bbox если нужно
            if self.show_bboxes.get() and detections:
                draw = ImageDraw.Draw(image)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()
                
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    label = det.get('label', 'unknown')
                    conf = det.get('confidence', 0)
                    
                    color = CLASS_COLORS.get(label, '#FFFFFF')
                    
                    # Рисуем прямоугольник
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Подпись
                    if self.show_labels.get():
                        if self.show_confidence.get():
                            text = f"{label} {conf:.0%}"
                        else:
                            text = label
                        
                        # Фон для текста
                        text_bbox = draw.textbbox((x1, y1 - 20), text, font=font)
                        draw.rectangle(text_bbox, fill=color)
                        draw.text((x1, y1 - 20), text, fill='black', font=font)
            
            # Масштабирование под canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Сохраняем пропорции
                img_ratio = image.width / image.height
                canvas_ratio = canvas_width / canvas_height
                
                if img_ratio > canvas_ratio:
                    new_width = canvas_width
                    new_height = int(canvas_width / img_ratio)
                else:
                    new_height = canvas_height
                    new_width = int(canvas_height * img_ratio)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Отображение
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.photo, anchor=tk.CENTER
            )
            
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text=f"Ошибка загрузки:\n{e}",
                fill='red', font=('Segoe UI', 12)
            )
    
    def refresh_image(self):
        """Перерисовка текущего изображения"""
        if self.annotations:
            annotation = self.annotations[self.current_index]
            self.load_and_display_image(
                annotation.get('image', ''),
                annotation.get('detections', [])
            )
    
    def next_image(self):
        """Следующее изображение"""
        if self.annotations and self.current_index < len(self.annotations) - 1:
            self.current_index += 1
            self.show_current_annotation()
    
    def prev_image(self):
        """Предыдущее изображение"""
        if self.annotations and self.current_index > 0:
            self.current_index -= 1
            self.show_current_annotation()
    
    def random_image(self):
        """Случайное изображение"""
        if self.annotations:
            self.current_index = random.randint(0, len(self.annotations) - 1)
            self.show_current_annotation()


def main():
    root = tk.Tk()
    app = AnnotationViewer(root)
    
    # Центрируем окно
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()

