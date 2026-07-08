"""
Модуль оценки резкости
"""

import numpy as np
import cv2


class SharpnessDetector:
    """Детектор резкости с выбором метода"""

    @staticmethod
    def rgb_to_gray(image):
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    @staticmethod
    def brenner(image):
        """Метод Бреннера"""
        gray = SharpnessDetector.rgb_to_gray(image)
        shifted = np.roll(gray, -2, axis=1)
        diff = (gray[:, :-2] - shifted[:, :-2]).astype(np.float32)
        diff = diff[diff > 0]
        brenner_score = np.sum(diff ** 2) / gray.size if len(diff) > 0 else 0
        return brenner_score

    @staticmethod
    def laplacian(image):
        """Дисперсия Лапласиана"""
        gray = SharpnessDetector.rgb_to_gray(image)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)

    @staticmethod
    def gradient(image):
        """Средний градиент"""
        gray = SharpnessDetector.rgb_to_gray(image)
        grad_x = np.abs(np.diff(gray.astype(np.float32), axis=1))
        grad_y = np.abs(np.diff(gray.astype(np.float32), axis=0))
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        return (np.mean(grad_x) + np.mean(grad_y)) / 2

    @staticmethod
    def hybrid(image):
        """Гибридный метод"""
        gray = SharpnessDetector.rgb_to_gray(image)

        # Бреннер
        shifted = np.roll(gray, -2, axis=1)
        diff = (gray[:, :-2] - shifted[:, :-2]).astype(np.float32)
        diff = diff[diff > 0]
        brenner_score = np.sum(diff ** 2) / gray.size if len(diff) > 0 else 0

        # Лапласиан
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)

        # Градиенты
        grad_x = np.abs(np.diff(gray.astype(np.float32), axis=1))
        grad_y = np.abs(np.diff(gray.astype(np.float32), axis=0))
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        mean_gradient = (np.mean(grad_x) + np.mean(grad_y)) / 2

        # Нормализация
        brenner_norm = min(100, brenner_score / 10)
        laplacian_norm = min(100, laplacian_var / 10)
        gradient_norm = min(100, mean_gradient * 5)

        if mean_gradient < 3:
            hybrid_score = max(10, min(30, gradient_norm))
        else:
            hybrid_score = brenner_norm * 0.2 + laplacian_norm * 0.2 + gradient_norm * 0.6

        if hybrid_score >= 65:
            return 100, "Резкое", hybrid_score
        elif hybrid_score >= 35:
            return 60, "Умеренное размытие", hybrid_score
        else:
            return 20, "Сильное размытие", hybrid_score

    @staticmethod
    def fft(image):
        """FFT-метод"""
        gray = SharpnessDetector.rgb_to_gray(image)
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        low_radius = min(h, w) // 8

        low_region = magnitude[center_h - low_radius:center_h + low_radius,
                     center_w - low_radius:center_w + low_radius]
        low_energy = np.sum(low_region)

        high_region = np.concatenate([
            magnitude[:low_radius, :].flatten(),
            magnitude[-low_radius:, :].flatten(),
            magnitude[:, :low_radius].flatten(),
            magnitude[:, -low_radius:].flatten()
        ])
        high_energy = np.sum(high_region)

        ratio = high_energy / (low_energy + 1e-6)
        score = min(100, ratio * 200)

        if ratio >= 0.15:
            return 100, "Резкое", score
        elif ratio >= 0.08:
            return 60, "Умеренное размытие", score
        else:
            return 20, "Сильное размытие", score