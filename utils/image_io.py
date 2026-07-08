"""
Модуль загрузки и сохранения изображений
"""

from PIL import Image
import numpy as np
import os


class ImageIO:
    """Ввод-вывод изображений"""

    @staticmethod
    def load(path):
        """Загрузка изображения из файла"""
        img = Image.open(path)
        if img.mode == 'RGB':
            return np.array(img)
        elif img.mode == 'L':
            return np.array(img)
        else:
            return np.array(img.convert('RGB'))

    @staticmethod
    def save(image, path):
        """Сохранение изображения"""
        if len(image.shape) == 3:
            img = Image.fromarray(image)
        else:
            img = Image.fromarray(image, mode='L')
        img.save(path)

    @staticmethod
    def get_supported_formats():
        return ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')