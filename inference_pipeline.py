# inference_pipeline.py (исправленная версия)
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# === Configuration ===
device = torch.device("cpu")
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "CP2.pth")

# Piece types and mappings
piece_types = ['K','Q','R','B','N','P','k','q','r','b','n','p','e']
idx_to_label = {i: label for i, label in enumerate(piece_types)}

# Inference transform (match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

# === Model definition ===
class ChessPieceCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(ChessPieceCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Глобальная переменная для модели
model = None

def load_model():
    """
    Загружает модель в память один раз.
    """
    global model
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}")
        
        model = ChessPieceCNN(len(piece_types)).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        
def process_image(image_path: str) -> str:
    """
    Принимает путь к обрезанному изображению шахматной доски, 
    запускает инференс и возвращает FEN-строку.
    """
    global model
    if model is None:
        raise RuntimeError("Модель не была загружена. Пожалуйста, вызовите load_model() перед использованием.")

    try:
        # 1) Загружаем уже обрезанную доску
        board_img_np = cv2.imread(image_path)
        if board_img_np is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")
            
        board_img = Image.fromarray(cv2.cvtColor(board_img_np, cv2.COLOR_BGR2RGB))
        
        # 2) Разбиваем на квадраты
        w, h = board_img.size
        sq_w, sq_h = w // 8, h // 8
        squares = []
        for r in range(8):
            for c in range(8):
                box = (c * sq_w, r * sq_h, (c + 1) * sq_w, (r + 1) * sq_h)
                squares.append(board_img.crop(box))

        # 3) Делаем предсказание
        inputs = torch.stack([transform(sq) for sq in squares]).to(device)
        with torch.no_grad():
            out = model(inputs)
            preds = out.argmax(dim=1).cpu().tolist()
            
        # =========================================================
        # ОТЛАДКА: Выводим предсказания, чтобы проверить, что модель видит
        print("Предсказания модели для 64 квадратов (0-63):")
        labels = [idx_to_label[p] for p in preds]
        print(labels)
        # =========================================================

        # 4) Собираем FEN с дополнительной проверкой
        fen_rows = []
        for r in range(8):
            row_str = ""
            empty_count = 0
            for c in range(8):
                idx = r * 8 + c
                lbl = idx_to_label[preds[idx]]
                
                if lbl == 'e':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += lbl
            
            if empty_count > 0:
                row_str += str(empty_count)
            
            # --- НОВАЯ ПРОВЕРКА ВАЛИДНОСТИ FEN-РЯДА ---
            # Эта проверка поможет выявить некорректные предсказания модели,
            # которые приводят к "нелегальным" FEN-строкам.
            total_squares = 0
            for char in row_str:
                if char.isdigit():
                    total_squares += int(char)
                else:
                    total_squares += 1
            
            if total_squares != 8:
                print(f"ОШИБКА FEN: Ряд '{row_str}' на позиции {r+1} имеет некорректную длину. Ожидается 8 клеток, получено {total_squares}.")
                raise ValueError(f"Некорректная FEN-строка: Ряд {r+1} имеет длину {total_squares}, а не 8.")

            fen_rows.append(row_str)
            # ----------------------------------------

        fen_placement = '/'.join(fen_rows)
        
        # --- Возвращаем стандартную FEN-строку с пробелами ---
        full_fen = f"{fen_placement} w - - 0 1"

        # =========================================================
        # ОТЛАДКА: Выводим итоговую FEN-строку
        print(f"Сгенерированная FEN-строка: {full_fen}")
        # =========================================================
        
        return full_fen
    except Exception as e:
        print(f"Ошибка в процессе обработки: {e}")
        raise e

if __name__ == '__main__':
    # Пример использования
    print("Запускаем пайплайн для предсказания FEN.")
    try:
        load_model()
        input_image_path = input("Введите путь к обрезанному изображению доски: ")
        fen_result = process_image(input_image_path)
        print(f"FEN-строка: {fen_result}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
