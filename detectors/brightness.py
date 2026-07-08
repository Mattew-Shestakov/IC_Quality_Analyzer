"""
Модуль оценки экспозиции
"""

import numpy as np


class BrightnessDetector:
    """Детектор экспозиции"""

    @staticmethod
    def rgb_to_gray(image):
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    @staticmethod
    def analyze(image):
        """Анализ экспозиции"""
        gray = BrightnessDetector.rgb_to_gray(image)
        mean = np.mean(gray)

        # Классификация
        if mean < 40:
            score = 20
            status = "Слишком темно"
        elif mean > 200:
            score = 20
            status = "Слишком светло"
        elif mean < 60 or mean > 180:
            score = 60
            status = "Приемлемо"
        else:
            score = 100
            status = "Норма"

        return {
            'mean': mean,
            'score': score,
            'status': status
        }