"""
Модуль генерации отчётов
"""

import csv
from datetime import datetime


class ReportGenerator:
    """Генератор отчётов"""

    @staticmethod
    def generate_text_report(analysis_result, filename):
        """Генерация текстового отчёта"""
        report = []
        report.append("=" * 60)
        report.append("ОТЧЁТ О КАЧЕСТВЕ ИЗОБРАЖЕНИЯ")
        report.append("=" * 60)
        report.append(f"Файл: {filename}")
        report.append("")
        report.append(f"Резкость: {analysis_result['sharpness_score']}/100 ({analysis_result['sharpness_status']})")
        report.append(f"Экспозиция: {analysis_result['exposure_score']}/100 ({analysis_result['exposure_status']})")
        report.append(f"Контраст: {analysis_result['contrast_score']}/100 ({analysis_result['contrast_status']})")
        report.append(f"Шум: {analysis_result['noise_score']}/100 ({analysis_result['noise_status']})")
        report.append("")
        report.append(f"Общая оценка: {analysis_result['overall_score']:.1f}/100")
        report.append(f"Вердикт: {analysis_result['verdict']}")
        report.append("=" * 60)
        return "\n".join(report)

    @staticmethod
    def export_csv(results, output_path):
        """Экспорт результатов в CSV"""
        if not results:
            return

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Файл", "Резкость(%)", "Экспозиция(%)", "Контраст(%)",
                "Шум(%)", "Общая(%)", "Вердикт", "Проблемы"
            ])
            for r in results:
                writer.writerow([
                    r['filename'],
                    r['sharpness_score'],
                    r['exposure_score'],
                    r['contrast_score'],
                    r['noise_score'],
                    r['overall_score'],
                    r['verdict'],
                    ", ".join(r['failed_criteria'])
                ])