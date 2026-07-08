"""
Модуль оценки уровня шума
"""

import numpy as np
from scipy.ndimage import median_filter


class NoiseDetector:
    """Детектор шума"""

    @staticmethod
    def rgb_to_gray(image):
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    @staticmethod
    def analyze(image):
        """Анализ уровня шума (SNR)"""
        gray = NoiseDetector.rgb_to_gray(image)

        # Медианная фильтрация
        filtered = median_filter(gray.astype(np.float32), size=3)
        filtered = filtered[:gray.shape[0], :gray.shape[1]]

        # Карта шума
        noise = np.abs(gray.astype(np.float32) - filtered)
        noise_level = np.mean(noise)

        # SNR
        signal = np.mean(gray)
        if noise_level > 0:
            snr = 20 * np.log10(signal / (noise_level + 1e-6))
        else:
            snr = 100

        if snr >= 40:
            score = 100
            status = "Низкий"
        elif snr >= 30:
            score = 60
            status = "Средний"
        else:
            score = 30
            status = "Высокий"

        return {
            'snr': snr,
            'noise_level': noise_level,
            'score': score,
            'status': status
        }