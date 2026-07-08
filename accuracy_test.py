"""
Расчёт точности и СКО по результатам пакетного анализа
"""

import pandas as pd
import numpy as np
from scipy import stats
import glob
import os


def calculate_accuracy_from_csv(sharp_csv, blurry_csv):
    """Расчёт точности по CSV-файлам из программы"""

    # Загружаем результаты
    sharp_df = pd.read_csv(sharp_csv)
    blurry_df = pd.read_csv(blurry_csv)

    # Резкие: ожидаем "ПРИГОДНО"
    sharp_correct = (sharp_df['Вердикт'] == "ПРИГОДНО").sum()
    sharp_total = len(sharp_df)
    sharp_accuracy = sharp_correct / sharp_total * 100

    # Размытые: ожидаем "НЕ ПРИГОДНО"
    blurry_correct = (blurry_df['Вердикт'] == "НЕ ПРИГОДНО").sum()
    blurry_total = len(blurry_df)
    blurry_accuracy = blurry_correct / blurry_total * 100

    # Общая точность
    total = sharp_total + blurry_total
    correct = sharp_correct + blurry_correct
    accuracy = correct / total * 100

    # Создаём массив ошибок (1 — ошибка, 0 — верно)
    errors = []

    for _, row in sharp_df.iterrows():
        errors.append(0 if row['Вердикт'] == "ПРИГОДНО" else 1)

    for _, row in blurry_df.iterrows():
        errors.append(0 if row['Вердикт'] == "НЕ ПРИГОДНО" else 1)

    errors = np.array(errors)
    error_rate = np.mean(errors) * 100
    std_error = np.std(errors) * 100

    # Доверительный интервал 95%
    n = len(errors)
    z = 1.96
    ci = z * std_error / np.sqrt(n)

    print("=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)

    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Резких изображений:    {sharp_total}")
    print(f"   Размытых изображений:  {blurry_total}")
    print(f"   Всего:                 {total}")

    print(f"\n📈 ТОЧНОСТЬ ПО ТИПАМ:")
    print(f"   Резкие:   {sharp_correct}/{sharp_total} = {sharp_accuracy:.1f}%")
    print(f"   Размытые: {blurry_correct}/{blurry_total} = {blurry_accuracy:.1f}%")

    print(f"\n🎯 ИТОГОВЫЕ МЕТРИКИ:")
    print(f"   Точность (Accuracy):      {accuracy:.2f}%")
    print(f"   Ошибка (Error rate):      {error_rate:.2f}%")
    print(f"   СКО (σ):                  {std_error:.2f}%")
    print(f"   Доверительный интервал 95%: ±{ci:.2f}%")

    errors_count = np.sum(errors)
    print(f"\n❌ ОШИБКИ КЛАССИФИКАЦИИ:")
    print(f"   Всего ошибок: {errors_count} из {total} ({error_rate:.1f}%)")

    return {
        'accuracy': accuracy,
        'std_error': std_error,
        'confidence_interval': ci,
        'sharp_accuracy': sharp_accuracy,
        'blurry_accuracy': blurry_accuracy,
        'errors_count': errors_count,
        'total': total,
        'correct': correct
    }


if __name__ == "__main__":
    # Укажи пути к твоим CSV-файлам
    sharp_csv = "sharp_results.csv"
    blurry_csv = "blurry_results.csv"

    if not os.path.exists(sharp_csv) or not os.path.exists(blurry_csv):
        print("❌ CSV-файлы не найдены!")
        print("   Сначала выполни пакетный анализ в программе и экспортируй CSV.")
        print("   Затем укажи правильные пути к файлам.")
    else:
        calculate_accuracy_from_csv(sharp_csv, blurry_csv)