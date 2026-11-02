# -*- coding: utf-8 -*-
import os
import glob
import time
import numpy as np
import matplotlib.image as mpimg

def audit_drawings(dataset_dir):
    pngs = glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True)
    total = len(pngs)
    print(f"Начало аудита {total} чертежей в {dataset_dir}...")

    empty_geometry = []
    suspicious_geometry = []
    corrupted_files = []
    valid_count = 0

    t0 = time.time()
    for idx, p in enumerate(pngs, 1):
        try:
            # Проверяем размер файла (пустой файл < 10KB)
            if os.path.getsize(p) < 10000:
                corrupted_files.append((p, "Размер файла слишком мал"))
                continue

            img = mpimg.imread(p)
            h, w = img.shape[:2]

            # Зоны трех проекций (без учета заголовков видов)
            # 1. Главный вид (front)
            roi_front = img[int(0.18 * h):int(0.54 * h), int(0.09 * w):int(0.42 * w)]
            # 2. Вид сверху (top)
            roi_top = img[int(0.66 * h):int(0.92 * h), int(0.09 * w):int(0.42 * w)]
            # 3. Вид слева (left)
            roi_left = img[int(0.18 * h):int(0.54 * h), int(0.47 * w):int(0.80 * w)]

            dark_front = np.sum(roi_front < 0.5)
            dark_top = np.sum(roi_top < 0.5)
            dark_left = np.sum(roi_left < 0.5)

            total_geom_dark = dark_front + dark_top + dark_left

            # Если во всех трех зонах почти нет линий (меньше 50 темных пикселей)
            if total_geom_dark < 50:
                empty_geometry.append((p, f"F:{dark_front}, T:{dark_top}, L:{dark_left}"))
            elif dark_front < 10 or dark_top < 10 or dark_left < 10:
                suspicious_geometry.append((p, f"F:{dark_front}, T:{dark_top}, L:{dark_left}"))
            else:
                valid_count += 1

        except Exception as e:
            corrupted_files.append((p, str(e)))

        if idx % 500 == 0 or idx == total:
            print(f"  Проверено {idx}/{total} ({idx/total*100:.1f}%) | Пустых геометрий: {len(empty_geometry)} | Подозрительных: {len(suspicious_geometry)}")

    dt = time.time() - t0
    print("\n" + "="*60)
    print(f"ИТОГИ АУДИТА (затрачено {dt:.1f} сек):")
    print(f"Всего проверено чертежей:     {total}")
    print(f"Полностью валидных чертежей:  {valid_count}")
    print(f"Чертежей БЕЗ картинок (пустые): {len(empty_geometry)}")
    print(f"Чертежей с неполными видами:  {len(suspicious_geometry)}")
    print(f"Поврежденных файлов:          {len(corrupted_files)}")
    print("="*60)

    # Сохраняем список проблемных файлов
    with open(r"f:\Трек\audit_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Аудит датасета: {dataset_dir}\n")
        f.write(f"Всего файлов: {total}\n")
        f.write(f"Пустых (без геометрии): {len(empty_geometry)}\n")
        f.write(f"С неполными проекциями: {len(suspicious_geometry)}\n\n")
        
        f.write("--- СПИСОК ПУСТЫХ ЧЕРТЕЖЕЙ (БЕЗ ГЕОМЕТРИИ) ---\n")
        for p, detail in empty_geometry:
            f.write(f"{p} [{detail}]\n")
            
        f.write("\n--- СПИСОК ЧЕРТЕЖЕЙ С НЕПОЛНЫМИ ПРОЕКЦИЯМИ ---\n")
        for p, detail in suspicious_geometry:
            f.write(f"{p} [{detail}]\n")

    return empty_geometry, suspicious_geometry, corrupted_files

if __name__ == "__main__":
    audit_drawings(r"f:\Трек\dataset_2d_drawings")
