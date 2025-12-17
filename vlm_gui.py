import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
import threading
import sys
import os
import random
import json
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


COLORS = {
    'bg_dark': '#0d1117',
    'bg_medium': '#161b22',
    'bg_light': '#21262d',
    'bg_card': '#1c2128',
    'accent': '#58a6ff',
    'accent_hover': '#79c0ff',
    'accent_green': '#3fb950',
    'accent_red': '#f85149',
    'accent_yellow': '#d29922',
    'accent_purple': '#a371f7',
    'text': '#e6edf3',
    'text_dim': '#8b949e',
    'text_muted': '#6e7681',
    'border': '#30363d',
    'glass': '#00ffaa',
    'plastic': '#ff6b6b',
    'metal': '#4ecdc4',
    'paper': '#ffe66d',
    'organic': '#95e1a3',
}

CLASS_EMOJI = {
    'glass': '🍾',
    'plastic': '🥤',
    'metal': '🥫',
    'paper': '📄',
    'organic': '🍎',
}

PROMPT_TEMPLATE = """Describe the garbage detected based on this CV output:
{input}
Description:"""


class DescriptionLLM:
    def __init__(self, model_path: str):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        config_path = Path(model_path) / "llm_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {
                "prompt_template": PROMPT_TEMPLATE,
                "max_input_length": 256,
                "max_output_length": 128,
                "num_beams": 4
            }
    
    def generate(self, cv_output: dict) -> str:
        prompt = self.config.get("prompt_template", PROMPT_TEMPLATE).format(input=json.dumps(cv_output))
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.config.get("max_input_length", 256), 
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.get("max_output_length", 128),
                num_beams=self.config.get("num_beams", 4),
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ModernButton(tk.Canvas):
    
    def __init__(self, parent, text, command, width=140, height=36, 
                 bg_color=None, fg_color='white', **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS['bg_medium'], highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.width = width
        self.height = height
        self.bg_color = bg_color or COLORS['accent']
        self.fg_color = fg_color
        self.is_hovered = False
        self.is_pressed = False
        
        self.draw_button()
        
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_press)
        self.bind('<ButtonRelease-1>', self.on_release)
    
    def draw_button(self):
        self.delete('all')
        
        if self.is_pressed:
            color = self._darken(self.bg_color, 0.8)
        elif self.is_hovered:
            color = self._lighten(self.bg_color, 1.15)
        else:
            color = self.bg_color
        
        r = 6
        self.create_rounded_rect(1, 1, self.width-1, self.height-1, r, fill=color, outline='')
        
        self.create_text(self.width//2, self.height//2, text=self.text,
                        fill=self.fg_color, font=('Segoe UI', 10, 'bold'))
    
    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _darken(self, hex_color, factor):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f'#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}'
    
    def _lighten(self, hex_color, factor):
        r = min(255, int(int(hex_color[1:3], 16) * factor))
        g = min(255, int(int(hex_color[3:5], 16) * factor))
        b = min(255, int(int(hex_color[5:7], 16) * factor))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def on_enter(self, e):
        self.is_hovered = True
        self.draw_button()
        self.config(cursor='hand2')
    
    def on_leave(self, e):
        self.is_hovered = False
        self.is_pressed = False
        self.draw_button()
    
    def on_press(self, e):
        self.is_pressed = True
        self.draw_button()
    
    def on_release(self, e):
        self.is_pressed = False
        self.draw_button()
        if self.is_hovered and self.command:
            self.command()


class VLMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🗑️ VLM Garbage Detection")
        self.root.geometry("1500x950")
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.minsize(1200, 800)
        
        self.vlm = None
        self.llm = None
        self.current_image = None
        self.current_image_path = None
        self.detections = []
        self.analysis = None
        self.image_folder = None
        self.folder_images = []
        self.current_index = 0
        
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_confidence = tk.BooleanVar(value=True)
        self.scene_enabled = tk.BooleanVar(value=True)
        
        self.setup_styles()
        self.setup_ui()
        self.load_vlm_async()
        
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Treeview',
                       background=COLORS['bg_light'],
                       foreground=COLORS['text'],
                       fieldbackground=COLORS['bg_light'],
                       font=('Segoe UI', 10),
                       rowheight=28)
        style.configure('Treeview.Heading',
                       background=COLORS['bg_dark'],
                       foreground=COLORS['text'],
                       font=('Segoe UI', 10, 'bold'))
        style.map('Treeview', background=[('selected', COLORS['accent'])])
        
        style.configure('Dark.TCheckbutton',
                       background=COLORS['bg_medium'],
                       foreground=COLORS['text'],
                       font=('Segoe UI', 10))
    
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        header = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        header.pack(fill=tk.X, pady=(0, 15))
        
        title = tk.Label(header, text="🗑️ VLM Garbage Detection System",
                        font=('Segoe UI', 20, 'bold'),
                        bg=COLORS['bg_dark'], fg=COLORS['text'])
        title.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header, text="⏳ Загрузка моделей...",
                                     font=('Segoe UI', 11),
                                     bg=COLORS['bg_dark'], fg=COLORS['accent_yellow'])
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        content = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        content.pack(fill=tk.BOTH, expand=True)
        
        left_panel = tk.Frame(content, bg=COLORS['bg_medium'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        toolbar = tk.Frame(left_panel, bg=COLORS['bg_medium'])
        toolbar.pack(fill=tk.X, padx=10, pady=10)
        
        ModernButton(toolbar, "📂 Открыть", self.open_image, width=100).pack(side=tk.LEFT, padx=3)
        ModernButton(toolbar, "📁 Папка", self.open_folder, width=100).pack(side=tk.LEFT, padx=3)
        ModernButton(toolbar, "🎲 Случайное", self.random_image, width=110).pack(side=tk.LEFT, padx=3)
        
        nav_frame = tk.Frame(toolbar, bg=COLORS['bg_medium'])
        nav_frame.pack(side=tk.RIGHT)
        
        ModernButton(nav_frame, "◀", self.prev_image, width=40, 
                    bg_color=COLORS['bg_light']).pack(side=tk.LEFT, padx=2)
        self.nav_label = tk.Label(nav_frame, text="0 / 0",
                                  font=('Segoe UI', 10),
                                  bg=COLORS['bg_medium'], fg=COLORS['text_dim'], width=10)
        self.nav_label.pack(side=tk.LEFT, padx=5)
        ModernButton(nav_frame, "▶", self.next_image, width=40,
                    bg_color=COLORS['bg_light']).pack(side=tk.LEFT, padx=2)
        
        canvas_frame = tk.Frame(left_panel, bg=COLORS['bg_light'], 
                               highlightthickness=2, highlightbackground=COLORS['border'])
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, bg=COLORS['bg_light'], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.create_text(400, 300, 
                               text="📷 Перетащите изображение сюда\n\nили нажмите 'Открыть'\n\n← → для навигации",
                               fill=COLORS['text_muted'], font=('Segoe UI', 14),
                               justify=tk.CENTER, tags='placeholder')
        
        self.filename_label = tk.Label(left_panel, text="",
                                       font=('Segoe UI', 10),
                                       bg=COLORS['bg_medium'], fg=COLORS['text_dim'])
        self.filename_label.pack(pady=(0, 10))
        
        right_panel = tk.Frame(content, bg=COLORS['bg_medium'], width=450)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)
        
        self._create_card(right_panel, "📝 Описание (LLM)", self._create_description_content)
        self._create_card(right_panel, "🎯 Детекции", self._create_detections_content, expand=True)
        self._create_card(right_panel, "🌍 Сцена", self._create_scene_content)
        self._create_card(right_panel, "⚙️ Отображение", self._create_options_content)
        self._create_card(right_panel, "💬 Вопрос", self._create_qa_content)
        
        save_frame = tk.Frame(right_panel, bg=COLORS['bg_medium'])
        save_frame.pack(fill=tk.X, padx=10, pady=10)
        ModernButton(save_frame, "💾 Сохранить результат", self.save_result,
                    width=200, bg_color=COLORS['accent_green']).pack()
        
        self._create_legend(right_panel)
        
        self.canvas.bind('<Configure>', lambda e: self.redraw_image())
        self._setup_drag_drop()
    
    def _create_card(self, parent, title, content_func, expand=False):
        card = tk.Frame(parent, bg=COLORS['bg_card'], 
                       highlightthickness=1, highlightbackground=COLORS['border'])
        card.pack(fill=tk.BOTH if expand else tk.X, expand=expand, padx=10, pady=5)
        
        header = tk.Label(card, text=title,
                         font=('Segoe UI', 11, 'bold'),
                         bg=COLORS['bg_card'], fg=COLORS['text'],
                         anchor='w')
        header.pack(fill=tk.X, padx=10, pady=(8, 5))
        
        content_frame = tk.Frame(card, bg=COLORS['bg_card'])
        content_frame.pack(fill=tk.BOTH, expand=expand, padx=10, pady=(0, 8))
        content_func(content_frame)
    
    def _create_description_content(self, parent):
        self.description_text = tk.Text(parent, height=3, wrap=tk.WORD,
                                        font=('Segoe UI', 11),
                                        bg=COLORS['bg_light'], fg=COLORS['text'],
                                        insertbackground=COLORS['text'],
                                        relief=tk.FLAT, padx=8, pady=8,
                                        state=tk.DISABLED)
        self.description_text.pack(fill=tk.X)
    
    def _create_detections_content(self, parent):
        columns = ('Класс', 'Уверенность', 'Размер')
        self.det_tree = ttk.Treeview(parent, columns=columns, show='headings', height=6)
        
        self.det_tree.heading('Класс', text='Класс')
        self.det_tree.heading('Уверенность', text='Уверенность')
        self.det_tree.heading('Размер', text='Размер')
        
        self.det_tree.column('Класс', width=120)
        self.det_tree.column('Уверенность', width=100)
        self.det_tree.column('Размер', width=100)
        
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.det_tree.yview)
        self.det_tree.configure(yscrollcommand=scrollbar.set)
        
        self.det_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.det_count_label = tk.Label(parent.master, text="Найдено: 0 объектов",
                                        font=('Segoe UI', 10),
                                        bg=COLORS['bg_card'], fg=COLORS['text_dim'])
        self.det_count_label.pack(anchor='e', padx=10, pady=(0, 5))
    
    def _create_scene_content(self, parent):
        self.scene_frame = tk.Frame(parent, bg=COLORS['bg_card'])
        self.scene_frame.pack(fill=tk.X)
        
        self.scene_label = tk.Label(self.scene_frame, text="—",
                                    font=('Segoe UI', 14, 'bold'),
                                    bg=COLORS['bg_card'], fg=COLORS['text'])
        self.scene_label.pack(side=tk.LEFT)
        
        self.scene_conf_label = tk.Label(self.scene_frame, text="",
                                         font=('Segoe UI', 10),
                                         bg=COLORS['bg_card'], fg=COLORS['text_dim'])
        self.scene_conf_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def _create_options_content(self, parent):
        row1 = tk.Frame(parent, bg=COLORS['bg_card'])
        row1.pack(fill=tk.X)
        
        row2 = tk.Frame(parent, bg=COLORS['bg_card'])
        row2.pack(fill=tk.X, pady=(5, 0))
        
        for text, var, cmd in [("Boxes", self.show_boxes, self.redraw_image),
                               ("Метки", self.show_labels, self.redraw_image),
                               ("Conf", self.show_confidence, self.redraw_image)]:
            cb = tk.Checkbutton(row1, text=text, variable=var,
                               bg=COLORS['bg_card'], fg=COLORS['text'],
                               selectcolor=COLORS['bg_light'],
                               activebackground=COLORS['bg_card'],
                               activeforeground=COLORS['text'],
                               font=('Segoe UI', 10),
                               command=cmd)
            cb.pack(side=tk.LEFT, padx=8)
        
        cb_scene = tk.Checkbutton(row2, text="🌍 Scene classifier", variable=self.scene_enabled,
                                  bg=COLORS['bg_card'], fg=COLORS['text'],
                                  selectcolor=COLORS['bg_light'],
                                  activebackground=COLORS['bg_card'],
                                  activeforeground=COLORS['text'],
                                  font=('Segoe UI', 10),
                                  command=self.on_scene_toggle)
        cb_scene.pack(side=tk.LEFT, padx=8)
    
    def _create_qa_content(self, parent):
        q_frame = tk.Frame(parent, bg=COLORS['bg_card'])
        q_frame.pack(fill=tk.X)
        
        self.question_entry = tk.Entry(q_frame, font=('Segoe UI', 11),
                                       bg=COLORS['bg_light'], fg=COLORS['text'],
                                       insertbackground=COLORS['text'], relief=tk.FLAT)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=6)
        self.question_entry.insert(0, "Is there any plastic?")
        self.question_entry.bind('<Return>', lambda e: self.ask_question())
        
        ModernButton(q_frame, "Ask", self.ask_question, width=60, height=32).pack(side=tk.RIGHT)
        
        self.answer_label = tk.Label(parent, text="",
                                     font=('Segoe UI', 11),
                                     bg=COLORS['bg_card'], fg=COLORS['accent_green'],
                                     wraplength=380, justify=tk.LEFT, anchor='w')
        self.answer_label.pack(fill=tk.X, pady=(8, 0))
    
    def _create_legend(self, parent):
        legend_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        
        for cls in ['glass', 'plastic', 'metal', 'paper', 'organic']:
            color = COLORS.get(cls, '#ffffff')
            emoji = CLASS_EMOJI.get(cls, '')
            lbl = tk.Label(legend_frame, text=f"{emoji} {cls}",
                          font=('Segoe UI', 9), fg=color,
                          bg=COLORS['bg_medium'])
            lbl.pack(side=tk.LEFT, padx=6)
    
    def _setup_drag_drop(self):
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind('<<Drop>>', self.on_drop)
        except Exception:
            pass
    
    def on_drop(self, event):
        file_path = event.data.strip('{}')
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            self.load_image(file_path)
    
    def on_scene_toggle(self):
        if self.current_image_path and self.vlm:
            self.load_image(self.current_image_path)
    
    def load_vlm_async(self):
        def load():
            try:
                from vlm_annotation.ensemble_detector import EnsembleDetector
                
                self.detector = EnsembleDetector(
                    yolo_model_path="models/yolo/yolov8x/best.pt",
                    detr_model_path="models/rt-detr/rt-detr-101/m",
                    detr_processor_path="models/rt-detr/rt-detr-101/p",
                    conf_threshold=0.5
                )
                
                self.scene_classifier = None
                scene_path = None
                if Path("models/scene_classifier_yolo.pt").exists():
                    scene_path = "models/scene_classifier_yolo.pt"
                elif Path("models/scene_classifier.pt").exists():
                    scene_path = "models/scene_classifier.pt"
                
                if scene_path:
                    if 'yolo' in scene_path.lower():
                        from train_scene_yolo import SceneClassifierYOLO
                        self.scene_classifier = SceneClassifierYOLO(scene_path)
                    else:
                        from train_scene_classifier import SceneClassifierInference
                        self.scene_classifier = SceneClassifierInference(scene_path)
                
                self.scene_type = "YOLO" if scene_path and 'yolo' in scene_path.lower() else "MobileNet"
                
                llm_path = Path("models/description_llm")
                if llm_path.exists():
                    self.llm = DescriptionLLM(str(llm_path))
                    self.llm_loaded = True
                else:
                    self.llm = None
                    self.llm_loaded = False
                
                self.vlm = True
                self.root.after(0, self.on_vlm_loaded)
                
            except Exception as e:
                import traceback
                error_msg = f"{e}\n\n{traceback.format_exc()}"
                print(f"❌ Ошибка загрузки VLM:\n{error_msg}")
                self.root.after(0, lambda err=str(e): self.on_vlm_error(err))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def on_vlm_loaded(self):
        scene_info = f" | Сцена: {getattr(self, 'scene_type', 'N/A')}"
        llm_info = " | LLM: ✅" if getattr(self, 'llm_loaded', False) else " | LLM: ❌"
        self.status_label.config(
            text=f"✅ Модели загружены{scene_info}{llm_info}",
            fg=COLORS['accent_green']
        )
    
    def on_vlm_error(self, error):
        self.status_label.config(
            text=f"❌ Ошибка: {error[:40]}...",
            fg=COLORS['accent_red']
        )
        messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить модели:\n{error}")
    
    def get_cv_output(self, image) -> dict:
        detections = self.detector.detect(image)
        
        detection_summary = [
            {"label": det["label"], "confidence": round(det["confidence"], 2)}
            for det in detections
        ]
        
        scene = {"class": "unknown", "confidence": 0.0}
        if self.scene_classifier and self.scene_enabled.get():
            scene_result = self.scene_classifier.predict(image)
            scene = {
                "class": scene_result["class"],
                "confidence": round(scene_result["confidence"], 2)
            }
        
        return {
            "detections": detection_summary,
            "scene": scene
        }
    
    def generate_description(self, cv_output: dict) -> str:
        if self.llm:
            return self.llm.generate(cv_output)
        else:
            return self._fallback_description(cv_output)
    
    def _fallback_description(self, cv_output: dict) -> str:
        detections = cv_output["detections"]
        scene = cv_output["scene"]
        
        counts = {}
        for det in detections:
            label = det["label"]
            counts[label] = counts.get(label, 0) + 1
        
        total = sum(counts.values())
        
        if total == 0:
            return "No garbage detected."
        
        items = [f"{count} {cls}" for cls, count in counts.items() if count > 0]
        garbage_str = "There is " + ", ".join(items)
        
        if scene["class"] != "unknown" and scene["confidence"] >= 0.8:
            preposition = "in" if scene["class"] == "marshy" else "on"
            return f"{garbage_str} {preposition} the {scene['class']}."
        
        return f"{garbage_str} detected."
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.folder_images = [Path(file_path)]
            self.current_index = 0
            self.load_image(file_path)
    
    def open_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder:
            self.image_folder = Path(folder)
            self.folder_images = sorted(
                list(self.image_folder.glob("*.jpg")) + 
                list(self.image_folder.glob("*.png")) +
                list(self.image_folder.glob("*.jpeg"))
            )
            if self.folder_images:
                self.current_index = 0
                self.load_image(str(self.folder_images[0]))
                self.update_nav_label()
            else:
                messagebox.showwarning("Пусто", "В папке нет изображений")
    
    def random_image(self):
        search_dirs = [
            Path("data/1206-data/valid"),
            Path("data/1206-data/test"),
        ]
        
        images = []
        for d in search_dirs:
            if d.exists():
                images.extend(list(d.glob("*.jpg")) + list(d.glob("*.png")))
        
        if images:
            img = random.choice(images)
            self.folder_images = images
            self.current_index = self.folder_images.index(img)
            self.load_image(str(img))
            self.update_nav_label()
        else:
            messagebox.showwarning("Нет изображений", "Датасет не найден")
    
    def prev_image(self):
        if self.folder_images and self.current_index > 0:
            self.current_index -= 1
            self.load_image(str(self.folder_images[self.current_index]))
            self.update_nav_label()
    
    def next_image(self):
        if self.folder_images and self.current_index < len(self.folder_images) - 1:
            self.current_index += 1
            self.load_image(str(self.folder_images[self.current_index]))
            self.update_nav_label()
    
    def update_nav_label(self):
        if self.folder_images:
            self.nav_label.config(text=f"{self.current_index + 1} / {len(self.folder_images)}")
        else:
            self.nav_label.config(text="0 / 0")
    
    def load_image(self, path):
        if self.vlm is None:
            messagebox.showwarning("Подождите", "Модели ещё загружаются...")
            return
        
        self.current_image_path = path
        self.current_image = Image.open(path).convert('RGB')
        self.filename_label.config(text=Path(path).name)
        
        self.status_label.config(text="⏳ Анализ...", fg=COLORS['accent_yellow'])
        self.root.update()
        
        try:
            raw_detections = self.detector.detect(self.current_image)
            self.detections = raw_detections
            
            counts = {}
            for det in raw_detections:
                label = det['label']
                counts[label] = counts.get(label, 0) + 1
            
            scene = {"class": "unknown", "confidence": 0.0}
            if self.scene_classifier and self.scene_enabled.get():
                scene = self.scene_classifier.predict(self.current_image)
            elif not self.scene_enabled.get():
                scene = {"class": "disabled", "confidence": 0.0}
            
            cv_output = self.get_cv_output(self.current_image)
            description = self.generate_description(cv_output)
            
            self.analysis = {
                'garbage': {
                    'detections': raw_detections,
                    'counts': counts,
                    'total': sum(counts.values())
                },
                'scene': scene,
                'description': description,
                'cv_output': cv_output
            }
            
            self.update_description(description)
            self.update_detections(self.detections)
            self.update_scene(scene)
            self.redraw_image()
            
            count = self.analysis['garbage']['total']
            self.status_label.config(
                text=f"✅ Найдено: {count} объект{'а' if 1 < count < 5 else 'ов' if count >= 5 or count == 0 else ''}",
                fg=COLORS['accent_green']
            )
            
        except Exception as e:
            self.status_label.config(text=f"❌ Ошибка: {e}", fg=COLORS['accent_red'])
            import traceback
            traceback.print_exc()
    
    def redraw_image(self):
        if self.current_image is None:
            return
        
        img = self.current_image.copy()
        draw = ImageDraw.Draw(img)
        
        if self.show_boxes.get():
            for det in self.detections:
                box = det['box']
                label = det['label']
                conf = det['confidence']
                color = COLORS.get(label, '#ffffff')
                
                for i in range(3):
                    draw.rectangle([box[0]-i, box[1]-i, box[2]+i, box[3]+i], 
                                  outline=color)
                
                if self.show_labels.get():
                    emoji = CLASS_EMOJI.get(label, '')
                    text = f"{emoji} {label}"
                    if self.show_confidence.get():
                        text += f" {conf:.0%}"
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", 14)
                    except:
                        font = ImageFont.load_default()
                    
                    text_bbox = draw.textbbox((box[0], box[1] - 22), text, font=font)
                    
                    padding = 3
                    draw.rectangle([
                        text_bbox[0] - padding,
                        text_bbox[1] - padding,
                        text_bbox[2] + padding,
                        text_bbox[3] + padding
                    ], fill=color)
                    
                    draw.text((box[0], box[1] - 22), text, fill='black', font=font)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 10 and canvas_height > 10:
            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height
            
            if img_ratio > canvas_ratio:
                new_width = canvas_width - 20
                new_height = int(new_width / img_ratio)
            else:
                new_height = canvas_height - 20
                new_width = int(new_height * img_ratio)
            
            if new_width > 0 and new_height > 0:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo, anchor=tk.CENTER
        )
    
    def update_description(self, text):
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete('1.0', tk.END)
        self.description_text.insert('1.0', text)
        self.description_text.config(state=tk.DISABLED)
    
    def update_detections(self, detections):
        for item in self.det_tree.get_children():
            self.det_tree.delete(item)
        
        for det in detections:
            box = det['box']
            w = int(box[2] - box[0])
            h = int(box[3] - box[1])
            emoji = CLASS_EMOJI.get(det['label'], '')
            
            self.det_tree.insert('', tk.END, values=(
                f"{emoji} {det['label']}",
                f"{det['confidence']:.1%}",
                f"{w}×{h}"
            ))
        
        count = len(detections)
        self.det_count_label.config(
            text=f"Найдено: {count} объект{'а' if 1 < count < 5 else 'ов' if count >= 5 or count == 0 else ''}"
        )
    
    def update_scene(self, scene):
        cls = scene['class']
        conf = scene['confidence']
        
        if cls == 'disabled':
            self.scene_label.config(text="⏸️ Отключено", fg=COLORS['text_muted'])
            self.scene_conf_label.config(text="", fg=COLORS['text_dim'])
            return
        
        scene_emoji = {'grass': '🌿', 'marshy': '🌾', 'rocky': '🪨', 'sandy': '🏖️'}
        emoji = scene_emoji.get(cls, '❓')
        
        if conf >= 0.8:
            self.scene_label.config(text=f"{emoji} {cls.upper()}", fg=COLORS['accent_green'])
            self.scene_conf_label.config(text=f"({conf:.0%})", fg=COLORS['text_dim'])
        else:
            self.scene_label.config(text=f"{emoji} {cls}", fg=COLORS['accent_yellow'])
            self.scene_conf_label.config(text=f"({conf:.0%}) — неуверенно", fg=COLORS['accent_yellow'])
    
    def ask_question(self):
        if self.vlm is None or self.current_image_path is None:
            return
        
        question = self.question_entry.get().strip()
        if not question:
            return
        
        answer = self._answer_question(question)
        self.answer_label.config(text=f"💡 {answer}")
    
    def _answer_question(self, question: str) -> str:
        q = question.lower()
        counts = self.analysis['garbage']['counts']
        total = sum(counts.values())
        scene = self.analysis['scene']
        
        scene_words = ['where', 'scene', 'surface', 'ground', 'location', 'grass', 'marshy', 'rocky', 'sandy']
        
        if any(word in q for word in scene_words):
            if not self.scene_enabled.get():
                return "Scene classifier is disabled."
            if scene['class'] != 'unknown' and scene['confidence'] >= 0.8:
                return f"The scene is: {scene['class']} ({scene['confidence']:.0%} confidence)."
            return "Scene classification uncertain."
        
        if "what" in q or "describe" in q:
            return self.analysis['description']
        
        if "how many" in q:
            for cls in ['glass', 'plastic', 'metal', 'paper', 'organic']:
                if cls in q:
                    c = counts.get(cls, 0)
                    return f"{c} {cls} object{'s' if c != 1 else ''} detected."
            return f"{total} garbage object{'s' if total != 1 else ''} detected in total."
        
        for cls in ['glass', 'plastic', 'metal', 'paper', 'organic']:
            if cls in q:
                c = counts.get(cls, 0)
                if c > 0:
                    return f"Yes, {c} {cls} object{'s' if c != 1 else ''} detected."
                else:
                    return f"No {cls} detected."
        
        return self.analysis['description']
    
    def save_result(self):
        if self.current_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        
        if file_path:
            img = self.current_image.copy()
            draw = ImageDraw.Draw(img)
            
            for det in self.detections:
                box = det['box']
                label = det['label']
                conf = det['confidence']
                color = COLORS.get(label, '#ffffff')
                
                for i in range(3):
                    draw.rectangle([box[0]-i, box[1]-i, box[2]+i, box[3]+i], outline=color)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                text = f"{label} {conf:.0%}"
                text_bbox = draw.textbbox((box[0], box[1] - 24), text, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
                draw.text((box[0], box[1] - 24), text, fill='black', font=font)
            
            try:
                desc_font = ImageFont.truetype("arial.ttf", 18)
            except:
                desc_font = ImageFont.load_default()
            
            desc = self.analysis['description'] if self.analysis else ""
            draw.rectangle([0, img.height - 40, img.width, img.height], fill='black')
            draw.text((10, img.height - 35), desc, fill='white', font=desc_font)
            
            img.save(file_path)
            messagebox.showinfo("Сохранено", f"Результат сохранён:\n{file_path}")


def main():
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        print("⚠️ tkinterdnd2 не установлен. Drag-n-drop недоступен.")
        print("   Установите: pip install tkinterdnd2")
        root = tk.Tk()
    
    app = VLMApp(root)
    
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
