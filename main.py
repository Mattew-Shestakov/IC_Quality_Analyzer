"""
Image Quality Analyzer (IQA) - Гибридный детектор резкости (ФИНАЛ)
ВКР УрФУ, Фт-420009, Шестаков М.А.

Основной метод: градиенты (80% веса) + Бреннер + Лапласиан
"""

import sys
import os
import numpy as np
from datetime import datetime
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox,
    QTabWidget, QMessageBox, QProgressBar, QStatusBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider,
    QFrame, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QColor


class ImageProcessor:
    """Оптимизированные реализации алгоритмов"""

    @staticmethod
    def load_image(path):
        img = Image.open(path)
        if img.mode == 'RGB':
            return np.array(img)
        elif img.mode == 'L':
            return np.array(img)
        else:
            return np.array(img.convert('RGB'))

    @staticmethod
    def rgb_to_gray(rgb_image):
        if len(rgb_image.shape) == 2:
            return rgb_image
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    @staticmethod
    def laplacian_fast(image):
        h, w = image.shape
        result = np.zeros((h, w), dtype=np.float32)
        result[1:-1, 1:-1] = (
            image[1:-1, 2:] + image[1:-1, :-2] +
            image[2:, 1:-1] + image[:-2, 1:-1] -
            4 * image[1:-1, 1:-1]
        )
        return result

    @staticmethod
    def sharpness_fft(image):
        """
        ОЦЕНКА РЕЗКОСТИ ЧЕРЕЗ АНАЛИЗ ФУРЬЕ-СПЕКТРА
        """
        gray = ImageProcessor.rgb_to_gray(image)

        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = gray.shape
        center_h, center_w = h // 2, w // 2

        low_radius = min(h, w) // 8
        low_region = magnitude[center_h-low_radius:center_h+low_radius,
                               center_w-low_radius:center_w+low_radius]
        low_energy = np.sum(low_region)

        high_region = np.concatenate([
            magnitude[:low_radius, :].flatten(),
            magnitude[-low_radius:, :].flatten(),
            magnitude[:, :low_radius].flatten(),
            magnitude[:, -low_radius:].flatten()
        ])
        high_energy = np.sum(high_region)

        if low_energy > 0:
            ratio = high_energy / low_energy
        else:
            ratio = 0

        score = min(100, ratio * 200)

        if ratio >= 0.15:
            return 100, "Резкое", score
        elif ratio >= 0.08:
            return 60, "Умеренное размытие", score
        else:
            return 20, "Сильное размытие", score

    @staticmethod
    def calculate_exposure(image):
        gray = ImageProcessor.rgb_to_gray(image)
        flat = gray.flatten()
        hist = np.bincount(flat, minlength=256)
        hist_percent = hist / gray.size * 100
        return {
            'mean': np.mean(flat),
            'overexposed': np.sum(hist_percent[240:]),
            'underexposed': np.sum(hist_percent[:15])
        }

    @staticmethod
    def calculate_contrast(image):
        gray = ImageProcessor.rgb_to_gray(image)
        return {
            'std': np.std(gray),
            'dynamic_range': np.max(gray) - np.min(gray)
        }

    @staticmethod
    def median_filter_fast(image, size=3):
        from numpy.lib.stride_tricks import sliding_window_view
        pad = size // 2
        padded = np.pad(image, pad, mode='edge')
        windows = sliding_window_view(padded, (size, size))
        result = np.median(windows, axis=(2, 3)).astype(np.uint8)
        if result.shape != image.shape:
            result = result[:image.shape[0], :image.shape[1]]
        return result

    @staticmethod
    def estimate_noise(image):
        gray = ImageProcessor.rgb_to_gray(image)
        filtered = ImageProcessor.median_filter_fast(gray, size=3)
        filtered = filtered[:gray.shape[0], :gray.shape[1]]
        noise = np.abs(gray.astype(np.float32) - filtered.astype(np.float32))
        noise_level = np.mean(noise)
        signal = np.mean(gray)
        snr = 20 * np.log10(signal / (noise_level + 1e-6)) if noise_level > 0 else 100
        return {'noise_level': noise_level, 'snr': snr}

    @staticmethod
    def get_score_and_status(value, thresholds, names):
        for threshold, (score, status) in zip(thresholds, names):
            if value >= threshold:
                return score, status
        return names[-1][0], names[-1][1]

    @staticmethod
    def analyze_single(image, thresholds):
        """Полный анализ одного изображения"""

        sharp_score, sharp_status, sharp_val = ImageProcessor.sharpness_fft(image)

        exposure = ImageProcessor.calculate_exposure(image)
        mean = exposure['mean']

        exp_score = 100
        exp_status = "Норма"
        if mean < thresholds['exp_dark']:
            exp_score = 20
            exp_status = "Слишком темно"
        elif mean > thresholds['exp_bright']:
            exp_score = 20
            exp_status = "Слишком светло"
        elif mean < thresholds['exp_dark_acceptable'] or mean > thresholds['exp_bright_acceptable']:
            exp_score = 60
            exp_status = "Приемлемо"

        contrast = ImageProcessor.calculate_contrast(image)
        cont_val = contrast['std']
        cont_score, cont_status = ImageProcessor.get_score_and_status(
            cont_val,
            [thresholds['cont_high'], thresholds['cont_mid']],
            [(100, "Хороший"), (60, "Удовлетворительный"), (30, "Низкий")]
        )

        noise = ImageProcessor.estimate_noise(image)
        snr = noise['snr']
        noise_score, noise_status = ImageProcessor.get_score_and_status(
            snr,
            [thresholds['noise_low'], thresholds['noise_mid']],
            [(100, "Низкий"), (60, "Средний"), (30, "Высокий")]
        )

        overall = (sharp_score + exp_score + cont_score + noise_score) / 4
        is_good = all(s >= 60 for s in [sharp_score, exp_score, cont_score, noise_score])
        verdict = "ПРИГОДНО" if is_good else "НЕ ПРИГОДНО"

        failed = []
        if sharp_score < 60:
            failed.append(f"Резкость ({sharp_status})")
        if exp_score < 60:
            failed.append(f"Экспозиция ({exp_status})")
        if cont_score < 60:
            failed.append(f"Контраст ({cont_status})")
        if noise_score < 60:
            failed.append(f"Шум ({noise_status})")

        return {
            'sharpness_value': sharp_val,
            'sharpness_score': sharp_score,
            'sharpness_status': sharp_status,
            'exposure_mean': exposure['mean'],
            'exposure_over': exposure['overexposed'],
            'exposure_under': exposure['underexposed'],
            'exposure_score': exp_score,
            'exposure_status': exp_status,
            'contrast_value': contrast['std'],
            'contrast_range': contrast['dynamic_range'],
            'contrast_score': cont_score,
            'contrast_status': cont_status,
            'noise_level': noise['noise_level'],
            'noise_snr': noise['snr'],
            'noise_score': noise_score,
            'noise_status': noise_status,
            'overall_score': overall,
            'verdict': verdict,
            'failed_criteria': failed
        }


class BatchAnalysisThread(QThread):
    progress = Signal(int, str, int)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, folder_path, thresholds):
        super().__init__()
        self.folder_path = folder_path
        self.thresholds = thresholds
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    def run(self):
        try:
            files = [f for f in os.listdir(self.folder_path)
                    if f.lower().endswith(self.supported_formats)]

            if not files:
                self.error.emit("В папке нет изображений")
                return

            results = []
            total = len(files)

            for i, filename in enumerate(files):
                filepath = os.path.join(self.folder_path, filename)

                try:
                    image = ImageProcessor.load_image(filepath)
                    analysis = ImageProcessor.analyze_single(image, self.thresholds)
                    analysis['filename'] = filename
                    results.append(analysis)
                except Exception as e:
                    results.append({'filename': filename, 'error': str(e), 'verdict': 'ОШИБКА'})

                percent = int((i + 1) / total * 100)
                self.progress.emit(percent, filename, i + 1)

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class ThresholdPanel(QWidget):
    thresholdsChanged = Signal(dict)

    def __init__(self):
        super().__init__()
        self.thresholds = {
            'sharp_high': 60,
            'sharp_mid': 30,
            'exp_dark_acceptable': 60,
            'exp_dark': 40,
            'exp_bright_acceptable': 180,
            'exp_bright': 220,
            'cont_high': 45,
            'cont_mid': 30,
            'noise_low': 40,
            'noise_mid': 30,
        }
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)

        slider_style = """
            QSlider { min-height: 20px; }
            QSlider::groove:horizontal { height: 6px; background: #34495e; border-radius: 3px; }
            QSlider::handle:horizontal { background: #3498db; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
            QSlider::sub-page:horizontal { background: #27ae60; border-radius: 3px; }
        """

        # Резкость
        sharp_group = QGroupBox("1. РЕЗКОСТЬ (Гибридный метод)")
        sharp_layout = QVBoxLayout()

        info_label = QLabel("ℹ️ Комбинирует градиенты (80%) + Бреннер (20%)")
        info_label.setStyleSheet("color: #7f8c8d; font-size: 9px;")
        sharp_layout.addWidget(info_label)

        high_layout = QVBoxLayout()
        high_label = QLabel("✅ Отлично (изображение резкое)")
        high_label.setStyleSheet("color: #27ae60;")
        high_layout.addWidget(high_label)

        high_slider_layout = QHBoxLayout()
        self.sharp_high_slider = QSlider(Qt.Horizontal)
        self.sharp_high_slider.setRange(0, 100)
        self.sharp_high_slider.setValue(self.thresholds['sharp_high'])
        self.sharp_high_slider.setSingleStep(5)
        self.sharp_high_slider.setStyleSheet(slider_style)
        self.sharp_high_slider.valueChanged.connect(self.update_thresholds)
        high_slider_layout.addWidget(self.sharp_high_slider)
        self.sharp_high_value = QLabel(f"{self.thresholds['sharp_high']}%")
        self.sharp_high_value.setFixedWidth(50)
        high_slider_layout.addWidget(self.sharp_high_value)
        high_layout.addLayout(high_slider_layout)
        sharp_layout.addLayout(high_layout)

        mid_layout = QVBoxLayout()
        mid_label = QLabel("⚠️ Удовлетворительно (лёгкое размытие)")
        mid_label.setStyleSheet("color: #f39c12;")
        mid_layout.addWidget(mid_label)

        mid_slider_layout = QHBoxLayout()
        self.sharp_mid_slider = QSlider(Qt.Horizontal)
        self.sharp_mid_slider.setRange(0, 100)
        self.sharp_mid_slider.setValue(self.thresholds['sharp_mid'])
        self.sharp_mid_slider.setSingleStep(5)
        self.sharp_mid_slider.setStyleSheet(slider_style)
        self.sharp_mid_slider.valueChanged.connect(self.update_thresholds)
        mid_slider_layout.addWidget(self.sharp_mid_slider)
        self.sharp_mid_value = QLabel(f"{self.thresholds['sharp_mid']}%")
        self.sharp_mid_value.setFixedWidth(50)
        mid_slider_layout.addWidget(self.sharp_mid_value)
        mid_layout.addLayout(mid_slider_layout)
        sharp_layout.addLayout(mid_layout)

        sharp_group.setLayout(sharp_layout)
        layout.addWidget(sharp_group)

        # Экспозиция
        exp_group = QGroupBox("2. ЭКСПОЗИЦИЯ (средняя яркость /255)")
        exp_layout = QVBoxLayout()

        normal_layout = QVBoxLayout()
        normal_label = QLabel("✅ Нормальная экспозиция")
        normal_label.setStyleSheet("color: #27ae60;")
        normal_layout.addWidget(normal_label)

        normal_range_layout = QHBoxLayout()
        normal_range_layout.addWidget(QLabel("от"))
        self.exp_dark_acc_slider = QSlider(Qt.Horizontal)
        self.exp_dark_acc_slider.setRange(20, 100)
        self.exp_dark_acc_slider.setValue(self.thresholds['exp_dark_acceptable'])
        self.exp_dark_acc_slider.setStyleSheet(slider_style)
        self.exp_dark_acc_slider.valueChanged.connect(self.update_thresholds)
        normal_range_layout.addWidget(self.exp_dark_acc_slider)
        self.exp_dark_acc_value = QLabel(f"{self.thresholds['exp_dark_acceptable']}")
        self.exp_dark_acc_value.setFixedWidth(40)
        normal_range_layout.addWidget(self.exp_dark_acc_value)
        normal_range_layout.addWidget(QLabel("до"))
        self.exp_bright_acc_slider = QSlider(Qt.Horizontal)
        self.exp_bright_acc_slider.setRange(100, 230)
        self.exp_bright_acc_slider.setValue(self.thresholds['exp_bright_acceptable'])
        self.exp_bright_acc_slider.setStyleSheet(slider_style)
        self.exp_bright_acc_slider.valueChanged.connect(self.update_thresholds)
        normal_range_layout.addWidget(self.exp_bright_acc_slider)
        self.exp_bright_acc_value = QLabel(f"{self.thresholds['exp_bright_acceptable']}")
        self.exp_bright_acc_value.setFixedWidth(40)
        normal_range_layout.addWidget(self.exp_bright_acc_value)
        normal_layout.addLayout(normal_range_layout)
        exp_layout.addLayout(normal_layout)

        critical_layout = QVBoxLayout()
        critical_label = QLabel("❌ Критично (пересвет/недосвет)")
        critical_label.setStyleSheet("color: #e74c3c;")
        critical_layout.addWidget(critical_label)

        critical_range_layout = QHBoxLayout()
        critical_range_layout.addWidget(QLabel("<"))
        self.exp_dark_slider = QSlider(Qt.Horizontal)
        self.exp_dark_slider.setRange(10, 60)
        self.exp_dark_slider.setValue(self.thresholds['exp_dark'])
        self.exp_dark_slider.setStyleSheet(slider_style)
        self.exp_dark_slider.valueChanged.connect(self.update_thresholds)
        critical_range_layout.addWidget(self.exp_dark_slider)
        self.exp_dark_value = QLabel(f"{self.thresholds['exp_dark']}")
        self.exp_dark_value.setFixedWidth(40)
        critical_range_layout.addWidget(self.exp_dark_value)
        critical_range_layout.addWidget(QLabel("или >"))
        self.exp_bright_slider = QSlider(Qt.Horizontal)
        self.exp_bright_slider.setRange(200, 250)
        self.exp_bright_slider.setValue(self.thresholds['exp_bright'])
        self.exp_bright_slider.setStyleSheet(slider_style)
        self.exp_bright_slider.valueChanged.connect(self.update_thresholds)
        critical_range_layout.addWidget(self.exp_bright_slider)
        self.exp_bright_value = QLabel(f"{self.thresholds['exp_bright']}")
        self.exp_bright_value.setFixedWidth(40)
        critical_range_layout.addWidget(self.exp_bright_value)
        critical_layout.addLayout(critical_range_layout)
        exp_layout.addLayout(critical_layout)

        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group)

        # Контраст
        cont_group = QGroupBox("3. КОНТРАСТ (СКО)")
        cont_layout = QVBoxLayout()

        good_cont_layout = QVBoxLayout()
        good_cont_label = QLabel("✅ Хороший контраст")
        good_cont_label.setStyleSheet("color: #27ae60;")
        good_cont_layout.addWidget(good_cont_label)

        good_cont_slider_layout = QHBoxLayout()
        self.cont_high_slider = QSlider(Qt.Horizontal)
        self.cont_high_slider.setRange(10, 80)
        self.cont_high_slider.setValue(self.thresholds['cont_high'])
        self.cont_high_slider.setStyleSheet(slider_style)
        self.cont_high_slider.valueChanged.connect(self.update_thresholds)
        good_cont_slider_layout.addWidget(self.cont_high_slider)
        self.cont_high_value = QLabel(f"{self.thresholds['cont_high']}")
        self.cont_high_value.setFixedWidth(50)
        good_cont_slider_layout.addWidget(self.cont_high_value)
        good_cont_layout.addLayout(good_cont_slider_layout)
        cont_layout.addLayout(good_cont_layout)

        mid_cont_layout = QVBoxLayout()
        mid_cont_label = QLabel("⚠️ Удовлетворительный контраст")
        mid_cont_label.setStyleSheet("color: #f39c12;")
        mid_cont_layout.addWidget(mid_cont_label)

        mid_cont_slider_layout = QHBoxLayout()
        self.cont_mid_slider = QSlider(Qt.Horizontal)
        self.cont_mid_slider.setRange(5, 50)
        self.cont_mid_slider.setValue(self.thresholds['cont_mid'])
        self.cont_mid_slider.setStyleSheet(slider_style)
        self.cont_mid_slider.valueChanged.connect(self.update_thresholds)
        mid_cont_slider_layout.addWidget(self.cont_mid_slider)
        self.cont_mid_value = QLabel(f"{self.thresholds['cont_mid']}")
        self.cont_mid_value.setFixedWidth(50)
        mid_cont_slider_layout.addWidget(self.cont_mid_value)
        mid_cont_layout.addLayout(mid_cont_slider_layout)
        cont_layout.addLayout(mid_cont_layout)

        cont_group.setLayout(cont_layout)
        layout.addWidget(cont_group)

        # Шум
        noise_group = QGroupBox("4. ШУМ (SNR, дБ)")
        noise_layout = QVBoxLayout()

        low_noise_layout = QVBoxLayout()
        low_noise_label = QLabel("✅ Низкий уровень шума")
        low_noise_label.setStyleSheet("color: #27ae60;")
        low_noise_layout.addWidget(low_noise_label)

        low_noise_slider_layout = QHBoxLayout()
        self.noise_low_slider = QSlider(Qt.Horizontal)
        self.noise_low_slider.setRange(15, 60)
        self.noise_low_slider.setValue(self.thresholds['noise_low'])
        self.noise_low_slider.setStyleSheet(slider_style)
        self.noise_low_slider.valueChanged.connect(self.update_thresholds)
        low_noise_slider_layout.addWidget(self.noise_low_slider)
        self.noise_low_value = QLabel(f"{self.thresholds['noise_low']} дБ")
        self.noise_low_value.setFixedWidth(50)
        low_noise_slider_layout.addWidget(self.noise_low_value)
        low_noise_layout.addLayout(low_noise_slider_layout)
        noise_layout.addLayout(low_noise_layout)

        mid_noise_layout = QVBoxLayout()
        mid_noise_label = QLabel("⚠️ Средний уровень шума")
        mid_noise_label.setStyleSheet("color: #f39c12;")
        mid_noise_layout.addWidget(mid_noise_label)

        mid_noise_slider_layout = QHBoxLayout()
        self.noise_mid_slider = QSlider(Qt.Horizontal)
        self.noise_mid_slider.setRange(10, 45)
        self.noise_mid_slider.setValue(self.thresholds['noise_mid'])
        self.noise_mid_slider.setStyleSheet(slider_style)
        self.noise_mid_slider.valueChanged.connect(self.update_thresholds)
        mid_noise_slider_layout.addWidget(self.noise_mid_slider)
        self.noise_mid_value = QLabel(f"{self.thresholds['noise_mid']} дБ")
        self.noise_mid_value.setFixedWidth(50)
        mid_noise_slider_layout.addWidget(self.noise_mid_value)
        mid_noise_layout.addLayout(mid_noise_slider_layout)
        noise_layout.addLayout(mid_noise_layout)

        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)

        reset_btn = QPushButton("🔄 Сбросить на стандартные значения")
        reset_btn.clicked.connect(self.reset_thresholds)
        reset_btn.setStyleSheet("background-color: #e67e22; margin-top: 10px; padding: 8px;")
        layout.addWidget(reset_btn)

        layout.addStretch()

        container.setLayout(layout)
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def update_thresholds(self):
        self.thresholds = {
            'sharp_high': self.sharp_high_slider.value(),
            'sharp_mid': self.sharp_mid_slider.value(),
            'exp_dark_acceptable': self.exp_dark_acc_slider.value(),
            'exp_dark': self.exp_dark_slider.value(),
            'exp_bright_acceptable': self.exp_bright_acc_slider.value(),
            'exp_bright': self.exp_bright_slider.value(),
            'cont_high': self.cont_high_slider.value(),
            'cont_mid': self.cont_mid_slider.value(),
            'noise_low': self.noise_low_slider.value(),
            'noise_mid': self.noise_mid_slider.value(),
        }
        self.sharp_high_value.setText(f"{self.thresholds['sharp_high']}%")
        self.sharp_mid_value.setText(f"{self.thresholds['sharp_mid']}%")
        self.exp_dark_acc_value.setText(f"{self.thresholds['exp_dark_acceptable']}")
        self.exp_bright_acc_value.setText(f"{self.thresholds['exp_bright_acceptable']}")
        self.exp_dark_value.setText(f"{self.thresholds['exp_dark']}")
        self.exp_bright_value.setText(f"{self.thresholds['exp_bright']}")
        self.cont_high_value.setText(f"{self.thresholds['cont_high']}")
        self.cont_mid_value.setText(f"{self.thresholds['cont_mid']}")
        self.noise_low_value.setText(f"{self.thresholds['noise_low']} дБ")
        self.noise_mid_value.setText(f"{self.thresholds['noise_mid']} дБ")
        self.thresholdsChanged.emit(self.thresholds)

    def reset_thresholds(self):
        self.sharp_high_slider.setValue(60)
        self.sharp_mid_slider.setValue(30)
        self.exp_dark_acc_slider.setValue(60)
        self.exp_bright_acc_slider.setValue(180)
        self.exp_dark_slider.setValue(40)
        self.exp_bright_slider.setValue(220)
        self.cont_high_slider.setValue(45)
        self.cont_mid_slider.setValue(30)
        self.noise_low_slider.setValue(40)
        self.noise_mid_slider.setValue(30)

    def get_thresholds(self):
        return self.thresholds


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.current_image = None
        self.current_analysis = None
        self.batch_results = []
        self.batch_thread = None
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        self.setWindowTitle("Image Quality Analyzer v3.0 - Гибридный метод")
        self.setGeometry(50, 50, 1400, 850)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel("Image Quality Analyzer v3.0")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #3498db;")
        left_layout.addWidget(title)

        subtitle = QLabel("Гибридный метод (градиенты + Бреннер)")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; margin-bottom: 10px; font-size: 10pt;")
        left_layout.addWidget(subtitle)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("📷 Загрузить изображение")
        self.btn_load.clicked.connect(self.load_image)
        btn_layout.addWidget(self.btn_load)

        self.btn_analyze = QPushButton("🔍 Анализировать")
        self.btn_analyze.clicked.connect(self.analyze_current)
        self.btn_analyze.setEnabled(False)
        btn_layout.addWidget(self.btn_analyze)
        left_layout.addLayout(btn_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #34495e; margin: 10px 0;")
        left_layout.addWidget(line)

        self.threshold_panel = ThresholdPanel()
        self.threshold_panel.thresholdsChanged.connect(self.on_thresholds_changed)
        left_layout.addWidget(self.threshold_panel, 1)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #34495e; margin: 10px 0;")
        left_layout.addWidget(line2)

        results_group = QGroupBox("Результаты анализа")
        results_layout = QVBoxLayout()
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setMinimumHeight(200)
        self.text_results.setStyleSheet("font-family: monospace; font-size: 9pt;")
        results_layout.addWidget(self.text_results)
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)

        self.verdict_label = QLabel("")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 8px;")
        left_layout.addWidget(self.verdict_label)

        self.btn_export = QPushButton("💾 Экспорт отчёта")
        self.btn_export.clicked.connect(self.export_report)
        self.btn_export.setEnabled(False)
        left_layout.addWidget(self.btn_export)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        self.tabs = QTabWidget()

        tab_img = QWidget()
        img_layout = QVBoxLayout(tab_img)
        self.image_label = QLabel("Загрузите изображение")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2c3e50; padding: 50px; font-size: 12pt;")
        img_layout.addWidget(self.image_label)
        self.tabs.addTab(tab_img, "Изображение")

        tab_batch = QWidget()
        batch_layout = QVBoxLayout(tab_batch)

        batch_control = QHBoxLayout()
        self.btn_select_folder = QPushButton("📁 Выбрать папку")
        self.btn_select_folder.clicked.connect(self.select_batch_folder)
        batch_control.addWidget(self.btn_select_folder)

        self.btn_start_batch = QPushButton("🚀 Пакетный анализ")
        self.btn_start_batch.clicked.connect(self.start_batch_analysis)
        self.btn_start_batch.setEnabled(False)
        batch_control.addWidget(self.btn_start_batch)

        self.btn_export_csv = QPushButton("📊 Экспорт CSV")
        self.btn_export_csv.clicked.connect(self.export_to_csv)
        self.btn_export_csv.setEnabled(False)
        batch_control.addWidget(self.btn_export_csv)

        batch_control.addStretch()
        batch_layout.addLayout(batch_control)

        self.folder_info = QLabel("Выберите папку")
        batch_layout.addWidget(self.folder_info)

        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        batch_layout.addWidget(self.batch_progress)

        self.batch_status = QLabel("")
        batch_layout.addWidget(self.batch_status)

        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(8)
        self.batch_table.setHorizontalHeaderLabels([
            "Файл", "Резкость", "Экспозиция", "Контраст", "Шум", "Оценка", "Вердикт", "Проблемы"
        ])
        batch_layout.addWidget(self.batch_table)

        self.stats_label = QLabel("")
        batch_layout.addWidget(self.stats_label)

        self.tabs.addTab(tab_batch, "Пакетный анализ")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_widget)

        splitter.setSizes([550, 850])
        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a252f; }
            QWidget { background-color: #1a252f; color: #ecf0f1; }
            QPushButton { background-color: #3498db; border: none; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #7f8c8d; }
            QGroupBox { border: 1px solid #34495e; border-radius: 5px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QTextEdit { background-color: #2c3e50; border: 1px solid #34495e; border-radius: 3px; }
            QTableWidget { background-color: #2c3e50; alternate-background-color: #34495e; }
            QHeaderView::section { background-color: #3498db; color: white; padding: 4px; }
            QTabWidget::pane { border: 1px solid #34495e; background-color: #2c3e50; }
            QTabBar::tab { background-color: #34495e; padding: 6px 15px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #3498db; }
            QProgressBar { border: 1px solid #34495e; border-radius: 3px; }
            QProgressBar::chunk { background-color: #27ae60; border-radius: 3px; }
            QScrollArea { border: none; background: transparent; }
        """)

    def display_image(self, image):
        try:
            h, w = image.shape[:2]
            max_size = 500
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                pil_img = Image.fromarray(image)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                display_img = np.array(pil_img)
            else:
                display_img = image

            if len(display_img.shape) == 3:
                h, w, ch = display_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                h, w = display_img.shape
                qt_img = QImage(display_img.data, w, h, w, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(qt_img)
            scaled = pixmap.scaled(550, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            self.image_label.setStyleSheet("padding: 0;")
        except Exception as e:
            self.image_label.setText(f"Ошибка: {str(e)[:50]}")

    def load_image(self):
        path = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "*.png *.jpg *.jpeg *.bmp")[0]
        if path:
            self.image_path = path
            self.current_image = ImageProcessor.load_image(path)
            self.display_image(self.current_image)
            self.status_bar.showMessage(f"Загружено: {os.path.basename(path)}")
            self.btn_analyze.setEnabled(True)
            self.analyze_current()

    def analyze_current(self):
        if self.current_image is None:
            return
        thresholds = self.threshold_panel.get_thresholds()
        self.current_analysis = ImageProcessor.analyze_single(self.current_image, thresholds)
        self.update_results_display()
        self.btn_export.setEnabled(True)
        self.status_bar.showMessage("Анализ завершён")

    def on_thresholds_changed(self, thresholds):
        if self.current_image is not None:
            self.current_analysis = ImageProcessor.analyze_single(self.current_image, thresholds)
            self.update_results_display()

    def update_results_display(self):
        if self.current_analysis is None:
            return

        result = self.current_analysis

        failed_text = ""
        if result['failed_criteria']:
            failed_text = "\n┌──────────────────────────────────────────────────────────────┐\n"
            failed_text += "│ ПРИЧИНЫ НЕПРОХОЖДЕНИЯ КОНТРОЛЯ                              │\n"
            failed_text += "├──────────────────────────────────────────────────────────────┤\n"
            for crit in result['failed_criteria']:
                failed_text += f"│   • {crit:<59}│\n"
            failed_text += "└──────────────────────────────────────────────────────────────┘\n"

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ОТЧЁТ О КАЧЕСТВЕ ИЗОБРАЖЕНИЯ                  ║
║                    (Гибридный метод)                            ║
╚══════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────┐
│ 1. РЕЗКОСТЬ (Градиенты 80% + Бреннер 20%)                       │
├──────────────────────────────────────────────────────────────────┤
│   Значение:      {result['sharpness_value']:.1f}%
│   Статус:        {result['sharpness_status']}
│   Оценка:        {result['sharpness_score']}/100
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ 2. ЭКСПОЗИЦИЯ (средняя яркость /255)                            │
├──────────────────────────────────────────────────────────────────┤
│   Средняя яркость:   {result['exposure_mean']:.1f}/255
│   Пересвет:          {result['exposure_over']:.1f}%
│   Недосвет:          {result['exposure_under']:.1f}%
│   Статус:            {result['exposure_status']}
│   Оценка:            {result['exposure_score']}/100
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ 3. КОНТРАСТ (СКО)                                               │
├──────────────────────────────────────────────────────────────────┤
│   Контраст (СКО):    {result['contrast_value']:.1f}
│   Диапазон:          {result['contrast_range']:.0f}
│   Статус:            {result['contrast_status']}
│   Оценка:            {result['contrast_score']}/100
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ 4. УРОВЕНЬ ШУМА (SNR, дБ)                                       │
├──────────────────────────────────────────────────────────────────┤
│   Уровень шума:      {result['noise_level']:.2f}
│   SNR:               {result['noise_snr']:.1f} дБ
│   Статус:            {result['noise_status']}
│   Оценка:            {result['noise_score']}/100
└──────────────────────────────────────────────────────────────────┘
{failed_text}
┌──────────────────────────────────────────────────────────────────┐
│ 5. ИТОГОВАЯ ОЦЕНКА                                               │
├──────────────────────────────────────────────────────────────────┤
│   Общая оценка:      {result['overall_score']:.1f}/100
│                                                                  │
│   ВЕРДИКТ:          {result['verdict']}
└──────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Анализ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.text_results.setText(report)

        color = "#27ae60" if result['verdict'] == "ПРИГОДНО" else "#e74c3c"
        self.verdict_label.setText(f"  {result['verdict']}  ")
        self.verdict_label.setStyleSheet(f"font-size: 16pt; font-weight: bold; padding: 8px; background-color: {color}; border-radius: 8px;")

    def select_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if folder:
            self.batch_folder = folder
            self.folder_info.setText(f"Папка: {folder}")
            self.btn_start_batch.setEnabled(True)

    def start_batch_analysis(self):
        if not hasattr(self, 'batch_folder'):
            QMessageBox.warning(self, "Ошибка", "Сначала выберите папку")
            return

        thresholds = self.threshold_panel.get_thresholds()

        self.btn_start_batch.setEnabled(False)
        self.btn_select_folder.setEnabled(False)
        self.btn_export_csv.setEnabled(False)
        self.batch_progress.setVisible(True)
        self.batch_progress.setValue(0)
        self.batch_table.setRowCount(0)
        self.batch_status.setText("Анализ...")

        self.batch_thread = BatchAnalysisThread(self.batch_folder, thresholds)
        self.batch_thread.progress.connect(self.update_batch_progress)
        self.batch_thread.finished.connect(self.on_batch_finished)
        self.batch_thread.error.connect(self.on_batch_error)
        self.batch_thread.start()

    def update_batch_progress(self, percent, filename, count):
        self.batch_progress.setValue(percent)
        self.batch_status.setText(f"Обработано: {count} | Текущий: {filename}")

    def on_batch_finished(self, results):
        self.batch_results = results
        self.batch_table.setRowCount(len(results))

        good_count = 0
        bad_count = 0

        for row, r in enumerate(results):
            if 'error' in r:
                self.batch_table.setItem(row, 0, QTableWidgetItem(r['filename']))
                self.batch_table.setItem(row, 6, QTableWidgetItem("ОШИБКА"))
                continue

            self.batch_table.setItem(row, 0, QTableWidgetItem(r['filename']))
            self.batch_table.setItem(row, 1, QTableWidgetItem(f"{r['sharpness_score']}%"))
            self.batch_table.setItem(row, 2, QTableWidgetItem(f"{r['exposure_score']}%"))
            self.batch_table.setItem(row, 3, QTableWidgetItem(f"{r['contrast_score']}%"))
            self.batch_table.setItem(row, 4, QTableWidgetItem(f"{r['noise_score']}%"))

            score_item = QTableWidgetItem(f"{r['overall_score']:.0f}%")
            if r['overall_score'] >= 80:
                score_item.setBackground(QColor(0, 120, 0))
            elif r['overall_score'] >= 60:
                score_item.setBackground(QColor(120, 120, 0))
            else:
                score_item.setBackground(QColor(120, 0, 0))
            self.batch_table.setItem(row, 5, score_item)

            verdict_item = QTableWidgetItem(r['verdict'])
            if r['verdict'] == "ПРИГОДНО":
                verdict_item.setForeground(QColor(0, 255, 0))
                good_count += 1
            else:
                verdict_item.setForeground(QColor(255, 100, 100))
                bad_count += 1
            self.batch_table.setItem(row, 6, verdict_item)

            problems = ", ".join(r['failed_criteria']) if r['failed_criteria'] else "Нет"
            self.batch_table.setItem(row, 7, QTableWidgetItem(problems))

        self.batch_table.resizeColumnsToContents()
        self.batch_status.setText(f"Анализ завершён! Годных: {good_count}, Бракованных: {bad_count}")
        self.stats_label.setText(f"Статистика: Всего: {len(results)} | ПРИГОДНО: {good_count} | НЕ ПРИГОДНО: {bad_count}")

        self.btn_start_batch.setEnabled(True)
        self.btn_select_folder.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self.batch_progress.setVisible(False)

    def on_batch_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка", error_msg)
        self.btn_start_batch.setEnabled(True)
        self.btn_select_folder.setEnabled(True)
        self.batch_progress.setVisible(False)

    def export_to_csv(self):
        if not self.batch_results:
            QMessageBox.warning(self, "Ошибка", "Нет результатов для экспорта")
            return

        path = QFileDialog.getSaveFileName(self, "Сохранить CSV",
            f"raw_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV (*.csv)")[0]

        if path:
            try:
                import csv
                with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Файл",
                        "Резкость(дисперсия)", "Резкость(оценка%)",
                        "Экспозиция(яркость)", "Экспозиция(оценка%)",
                        "Контраст(СКО)", "Контраст(оценка%)",
                        "Шум(SNR_дБ)", "Шум(оценка%)",
                        "Общая(%)", "Вердикт", "Проблемы"
                    ])
                    for r in self.batch_results:
                        if 'error' in r:
                            writer.writerow([r['filename'], "ОШИБКА", "", "", "", "", "", "", "", "", "ОШИБКА", ""])
                        else:
                            writer.writerow([
                                r['filename'],
                                f"{r['sharpness_value']:.2f}", f"{r['sharpness_score']}",
                                f"{r['exposure_mean']:.1f}", f"{r['exposure_score']}",
                                f"{r['contrast_value']:.2f}", f"{r['contrast_score']}",
                                f"{r['noise_snr']:.1f}", f"{r['noise_score']}",
                                f"{r['overall_score']:.0f}",
                                r['verdict'], ", ".join(r['failed_criteria'])
                            ])
                QMessageBox.information(self, "Успех", f"CSV сохранён:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

    def export_report(self):
        if not self.text_results.toPlainText():
            return
        path = QFileDialog.getSaveFileName(self, "Сохранить отчёт",
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "*.txt")[0]
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.text_results.toPlainText())
            QMessageBox.information(self, "Успех", f"Отчёт сохранён:\n{path}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()