from flask import Flask, request, jsonify
import os
import datetime
import traceback
import cv2
import numpy as np

# --- Настройки ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# --- Импорты ---
try:
    import corner  # используем corner.py вместо pts
    print("✅ corner импортирован успешно")
except Exception as e:
    print(f"❌ Ошибка при импорте corner: {e}")

try:
    import inference_pipeline
    print("✅ inference_pipeline импортирован успешно")
except Exception as e:
    print(f"❌ Ошибка при импорте inference_pipeline: {e}")

# --- Загрузка модели ---
try:
    print("Загрузка модели CNN...")
    inference_pipeline.load_model()
    print("✅ Модель успешно загружена.")
except Exception as e:
    print(f"❌ Ошибка при загрузке модели: {e}")

# --- Вспомогательные функции ---
def crop_to_80_percent_width(image_path):
    """Обрезает изображение по центру, оставляя 80% ширины."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    h, w = img.shape[:2]
    new_w = int(w * 0.8)  # оставляем 80% ширины
    start_x = (w - new_w) // 2
    cropped = img[:, start_x:start_x + new_w]

    cropped_path = os.path.join(
        os.path.dirname(image_path),
        f"pre_crop_{os.path.basename(image_path)}"
    )
    cv2.imwrite(cropped_path, cropped)
    return cropped_path

# --- API ---
@app.route('/api/get_fen', methods=['POST'])
def get_fen():
    print("[DEBUG] Запрос получен")

    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    if not file or file.filename.strip() == "":
        return jsonify({"error": "Файл не получен"}), 400

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        ext = '.jpg'

    filename = os.path.join(
        UPLOAD_FOLDER,
        f"original_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}{ext}"
    )

    # Сохраняем оригинал
    image_bytes = file.read()
    with open(filename, 'wb') as f:
        f.write(image_bytes)
    print(f"[DEBUG] Сохранено исходное изображение ({len(image_bytes)} байт): {filename}")

    try:
        # Предварительное обрезание до 80% ширины
        pre_cropped_path = crop_to_80_percent_width(filename)
        print(f"[DEBUG] Предварительное обрезание сохранено: {pre_cropped_path}")

        # Поиск и вырезка доски через corner.py
        cropped_image_path = corner.crop_chessboard(pre_cropped_path)
        if cropped_image_path is None:
            return jsonify({'error': 'Не удалось найти доску на изображении.'}), 400

        print(f"[DEBUG] Обрезанное и выровненное изображение: {cropped_image_path}")

        # Обработка моделью
        fen = inference_pipeline.process_image(cropped_image_path)
        print(f"[DEBUG] Результат FEN: {fen}")

        return jsonify({'fen': fen})

    except Exception as e:
        print("[ERROR] Исключение в /api/get_fen:")
        traceback.print_exc()
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/ping')
def ping():
    return "pong"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
