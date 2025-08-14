import cv2
import numpy as np
import os

def order_points(pts):
    """Упорядочиваем точки по порядку: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def crop_chessboard(image_path):
    """
    Загружает изображение, убирает шум, находит шахматную доску и возвращает путь к выровненному изображению.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return None

    # 1. Шумоподавление
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 2. Преобразование в серый и размытие
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Детектирование краев
    edges = cv2.Canny(blur, 50, 150)

    # 4. Поиск контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Нахождение предполагаемой шахматной доски
    board_contour = None
    max_area = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                board_contour = approx

    if board_contour is None:
        print("❌ Шахматная доска не найдена.")
        return None

    # 6. Выровнять доску (перспективное преобразование)
    pts = board_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(denoised, M, (maxWidth, maxHeight))

    # 7. Сохранение результата
    output_path = os.path.join(os.path.dirname(image_path), "board_aligned.jpg")
    cv2.imwrite(output_path, warped)
    print(f"✅ Шахматная доска выровнена и сохранена: {output_path}")

    return output_path
