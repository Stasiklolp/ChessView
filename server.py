from flask import Flask, request, jsonify
import traceback
import cv2
import numpy as np
import corner
import inference_pipeline

app = Flask(__name__)

# --- Загрузка модели ---
try:
    print("Загрузка модели CNN...")
    inference_pipeline.load_model()
    print("✅ Модель успешно загружена.")
except Exception as e:
    print(f"❌ Ошибка при загрузке модели: {e}")

# --- Вспомогательная функция ---
def crop_to_80_percent_width(img):
    """Обрезает изображение по центру, оставляя 80% ширины."""
    h, w = img.shape[:2]
    new_w = int(w * 0.8)
    start_x = (w - new_w) // 2
    return img[:, start_x:start_x + new_w]

# --- API ---
@app.route('/api/get_fen', methods=['POST'])
def get_fen():
    print("[DEBUG] Запрос получен")

    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    if not file or file.filename.strip() == "":
        return jsonify({"error": "Файл не получен"}), 400

    try:
        # Чтение изображения в памяти
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Не удалось прочитать изображение'}), 400

        # Предварительное обрезание до 80% ширины
        img_cropped = crop_to_80_percent_width(img)

        # Поиск и вырезка доски через corner.py
        cropped_image = corner.crop_chessboard(img_cropped)
        if cropped_image is None:
            return jsonify({'error': 'Не удалось найти доску на изображении.'}), 400

        # Обработка моделью
        fen = inference_pipeline.process_image(cropped_image)
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
