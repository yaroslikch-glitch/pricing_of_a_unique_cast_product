# -*- coding: utf-8 -*-
"""
Скрипт очистки датасета YOLO от крепежа (болты, гайки, заклепки, шпильки)
и текстовых страниц из методичек с водяными знаками.
Перемещает отфильтрованные файлы в архивную папку dataset_yolo_trash_archive.
"""

import os
import shutil
from pathlib import Path

FASTENERS_KEYWORDS = [
    'cap_nuts', 'castle_nuts', 'conventional_rivets', 'convex_washer',
    'cylindrical_pins', 'eye_screws', 'flange_nut', 'grooved_pins',
    'hexagonal_nuts', 'keys_and_keyways', 'knob', 'locknuts',
    'lockwashers', 'roll_pins', 'screws_and_bolts', 'setscrew',
    'snap_rings', 'split_pins', 'square_nuts', 'studs', 't-nut',
    'taper_pins', 'tapping_screws', 'threaded_rods', 'thrust_washers',
    'washer_bolt'
]

# Страницы методичек, содержащие сплошной текст, таблицы или водяные знаки вместо чертежей
BAD_BOOK_PAGES = [
    'CHertezhi-detalej-mashin_p017', 'CHertezhi-detalej-mashin_p018',
    'CHertezhi-detalej-mashin_p019', 'CHertezhi-detalej-mashin_p020',
    'CHertezhi-detalej-mashin_p022', 'CHertezhi-detalej-mashin_p023',
    'CHertezhi-detalej-mashin_p024', 'CHertezhi-detalej-mashin_p025',
    'CHertezhi-detalej-mashin_p026', 'Detali_Mashin_i_Osnovy_Konstruirovaniya',
    'RSVPU_2017_740', 'SUSU_DVS_Dvigateli', 'SUSU_Metodichka',
    'Tekhnologiya_Liteynogo_Proizvodstva_Otlivki', 'Tavtilov',
    'design-of-foundry-tooling'
]

def clean_split(split_name="train", archive_dir="dataset_yolo_trash_archive"):
    img_dir = Path("dataset_yolo") / "images" / split_name
    lbl_dir = Path("dataset_yolo") / "labels" / split_name
    arch_img = Path(archive_dir) / "images" / split_name
    arch_lbl = Path(archive_dir) / "labels" / split_name
    
    arch_img.mkdir(parents=True, exist_ok=True)
    arch_lbl.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"Папка {img_dir} не найдена.")
        return 0, 0

    removed = 0
    kept = 0

    for img_path in list(img_dir.glob("*.png")):
        name_low = img_path.name.lower()
        stem = img_path.stem

        is_junk = False
        # Проверка 1: Крепеж
        if any(k in name_low for k in FASTENERS_KEYWORDS):
            is_junk = True
        # Проверка 2: Текстовые страницы методичек
        elif any(bp in stem for bp in BAD_BOOK_PAGES):
            # Сохраняем только p010..p013 из CHertezhi (где реальные чертежи валов/фланцев)
            if "CHertezhi-detalej-mashin_p01" in stem and any(f"p01{i}" in stem for i in [0, 1, 2, 3]):
                is_junk = False
            else:
                is_junk = True

        txt_path = lbl_dir / f"{stem}.txt"

        if is_junk:
            shutil.move(str(img_path), str(arch_img / img_path.name))
            if txt_path.exists():
                shutil.move(str(txt_path), str(arch_lbl / txt_path.name))
            removed += 1
        else:
            kept += 1

    print(f"[{split_name.upper()}] Удалено мусорных файлов: {removed} | Оставлено чистых чертежей: {kept}")
    return removed, kept

def main():
    print("=== Очистка датасета YOLO от крепежа и текстовых страниц ===")
    t_rem, t_kept = clean_split("train")
    v_rem, v_kept = clean_split("val")
    print("=" * 60)
    print(f"Всего перемещено в архив: {t_rem + v_rem} файлов.")
    print(f"Осталось в чистом датасете: {t_kept + v_kept} файлов (Train: {t_kept}, Val: {v_kept}).")

if __name__ == "__main__":
    main()
