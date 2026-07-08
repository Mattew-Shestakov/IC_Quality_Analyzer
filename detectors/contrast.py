"""
Модуль оценки контраста
"""

import numpy as np


class ContrastDetector:
    """Детектор контраста"""

    @staticmethod
    def rgb_to_gray(image):
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    @staticmethod
    def analyze(image):
        """Анализ контраста"""
        gray = ContrastDetector.rgb_to_gray(image)
        std = np.std(gray)
        dynamic_range = np.max(gray) - np.min(gray)

        if std >= 45 and dynamic_range >= 150:
            score = 100
            status = "Хороший"
        elif std >= 30 and dynamic_range >= 100:
            score = 60
            status = "Удовлетворительный"
        else:
            score = 30
            status = "Низкий"

        return {
            'std': std,
            'dynamic_range': dynamic_range,
            'score': score,
            'status': status
        }