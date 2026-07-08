"""
Создание тестовых изображений для проверки точности классификации
"""

import cv2
import numpy as np
import os
import random
from datetime import datetime

def create_test_dataset(size=256, count=500, output_dir="test_dataset_1000"):
    """
    Создаёт резкие и размытые изображения

    Аргументы:
        size: размер изображения (px)
        count: количество резких (и столько же размытых)
        output_dir: папка для сохранения
    """

    # Создаём папки
    sharp_dir = os.path.join(output_dir, "sharp")
    blurry_dir = os.path.join(output_dir, "blurry")
    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(blurry_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"СОЗДАНИЕ {count} РЕЗКИХ И {count} РАЗМЫТЫХ ИЗОБРАЖЕНИЙ")
    print(f"{'='*50}\n")

    start_time = datetime.now()

    for i in range(count):
        # Создаём резкое изображение
        img = np.zeros((size, size), dtype=np.uint8)
        bg = random.randint(30, 100)
        img[:, :] = bg

        # Случайные геометрические фигуры
        num_shapes = random.randint(3, 8)
        for _ in range(num_shapes):
            shape_type = random.choice(['rect', 'circle', 'line'])
            color = random.randint(150, 255)

            if shape_type == 'rect':
                x1 = random.randint(20, size - 80)
                y1 = random.randint(20, size - 80)
                w = random.randint(30, 100)
                h = random.randint(30, 100)
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, -1)

            elif shape_type == 'circle':
                x = random.randint(50, size - 50)
                y = random.randint(50, size - 50)
                r = random.randint(15, 50)
                cv2.circle(img, (x, y), r, color, -1)

            else:  # line
                x1 = random.randint(10, size - 10)
                y1 = random.randint(10, size - 10)
                x2 = random.randint(10, size - 10)
                y2 = random.randint(10, size - 10)
                cv2.line(img, (x1, y1), (x2, y2), color, random.randint(2, 5))

        # Добавляем мелкие детали (текстуру)
        for _ in range(300):
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            img[y, x] = random.randint(200, 255)

        # Сохраняем резкое
        cv2.imwrite(os.path.join(sharp_dir, f"sharp_{i:05d}.png"), img)

        # Создаём размытое (на основе этого же изображения)
        sigma = random.uniform(2.0, 5.0)
        blurry = cv2.GaussianBlur(img, (0, 0), sigma)
        cv2.imwrite(os.path.join(blurry_dir, f"blurry_{i:05d}.png"), blurry)

        # Прогресс
        if (i + 1) % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  Создано: {i+1}/{count} резких | {i+1}/{count} размытых | Время: {elapsed:.1f}с")

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*50}")
    print(f"✅ ГОТОВО!")
    print(f"{'='*50}")
    print(f"  Резких изображений:  {count}")
    print(f"  Размытых изображений: {count}")
    print(f"  Всего:               {count*2}")
    print(f"  Время создания:      {elapsed:.1f} сек")
    print(f"  Папка:               {output_dir}/")
    print(f"{'='*50}")

if __name__ == "__main__":
    create_test_dataset(count=500)