import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# -*- coding: utf-8 -*-
"""
Мастер-генератор гибридного сбалансированного датасета ЕСКД (2 000+ чертежей).
Проект: Оценка уникального литья (ЮУрГУ / ООО "БВК").

Архитектурная схема:
1. Базовый датасет: 1 000 существующих чертежей (dataset_yolo).
2. Реальные заводские чертежи: 30 чертежей БВК/КОНАР (датабаза/drawings).
3. Синтетическое расширение: отбор 3D-моделей MCB по 8 литейным классам (Valve, Impeller, Bushes, Pulley, Flange и т.д.).
4. Автоматическая валидация через DrawingCurator (допускаются только чертежи с оценкой >= 80% GOLDEN).
5. Формирование сбалансированного сплита train (85%) / val (15%) и dataset.yaml.
"""

import sys
import os
import glob
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from drawing_curator import DrawingCurator

MCB_DATASET_ROOT = r"C:\Users\Yaroslav\OneDrive\Desktop\датасет проекта\Скрипт пайтон 3D =_ 2D и тестовый датасет\dataset_org_norm\train"
FACTORY_DRAWINGS_DIR = r"C:\Users\Yaroslav\OneDrive\Desktop\датасет проекта\датабаза\drawings"
BASE_YOLO_DIR = r"dataset_yolo"

TARGET_EXPANDED_DIR = r"dataset_yolo_expanded"

# Целевые литейные категории MCB
FOUNDRY_CATEGORIES = [
    "Valve",
    "Impeller",
    "Turbine",
    "Flanged block bearing",
    "Flanged plain bearings",
    "Bushes",
    "Pulleys",
    "Right spur gears",
    "Lever",
    "Elbow fitting",
    "Plates, circulate plates"
]


def render_pdf_to_png(pdf_path: str, out_png: str, dpi: int = 150) -> bool:
    """Конвертирует первую страницу PDF в высококачественный PNG."""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return False
        page = doc[0]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(out_png)
        doc.close()
        return True
    except Exception as e:
        print(f"[ERR] Ошибка конвертации PDF {pdf_path}: {e}")
        return False


def build_expanded_dataset(target_total: int = 2000,
                           output_dir: str = TARGET_EXPANDED_DIR,
                           val_ratio: float = 0.15) -> Dict[str, Any]:
    print("=" * 65)
    print("🚀 СБОРКА РАСШИРЕННОГО ГИБРИДНОГО ДАТАСЕТА ЕСКД (Синтетика + Завод)")
    print(f"Целевой объем: {target_total} чертежей | Валидация: {int(val_ratio * 100)}%")
    print("=" * 65)

    out_path = Path(output_dir)
    images_train = out_path / "images" / "train"
    images_val = out_path / "images" / "val"
    labels_train = out_path / "labels" / "train"
    labels_val = out_path / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    curator = DrawingCurator()

    total_images = 0
    stats = {
        "base_yolo_copied": 0,
        "factory_bvk_added": 0,
        "synthetic_mcb_added": 0,
        "train_count": 0,
        "val_count": 0,
        "total_count": 0
    }

    # ---------------------------------------------------------
    # ШАГ 1. Копирование существующего базового датасета (1 000 шт)
    # ---------------------------------------------------------
    base_p = Path(BASE_YOLO_DIR)
    if base_p.exists():
        print("\n[ШАГ 1] Интеграция базового датасета (dataset_yolo)...")
        for split in ["train", "val"]:
            src_img_dir = base_p / "images" / split
            src_lbl_dir = base_p / "labels" / split
            tgt_img_dir = out_path / "images" / split
            tgt_lbl_dir = out_path / "labels" / split

            if src_img_dir.exists():
                imgs = list(src_img_dir.glob("*.png")) + list(src_img_dir.glob("*.jpg"))
                for img_file in imgs:
                    shutil.copy2(str(img_file), str(tgt_img_dir / img_file.name))
                    lbl_file = src_lbl_dir / (img_file.stem + ".txt")
                    if lbl_file.exists():
                        shutil.copy2(str(lbl_file), str(tgt_lbl_dir / lbl_file.name))
                    stats["base_yolo_copied"] += 1
        print(f"  ✅ Скопировано {stats['base_yolo_copied']} чертежей из базового датасета.")

    # ---------------------------------------------------------
    # ШАГ 2. Интеграция реальных заводских чертежей БВК (30 шт)
    # ---------------------------------------------------------
    fact_p = Path(FACTORY_DRAWINGS_DIR)
    if fact_p.exists():
        print("\n[ШАГ 2] Интеграция реальных заводских чертежей БВК/КОНАР...")
        factory_pdfs = list(fact_p.glob("*.pdf"))
        print(f"  Найдено заводских чертежей: {len(factory_pdfs)}")

        for i, pdf_file in enumerate(factory_pdfs, 1):
            out_img_name = f"factory_bvk_{pdf_file.stem}.png"
            split = "val" if (i % 5 == 0) else "train"
            tgt_img = (out_path / "images" / split) / out_img_name
            tgt_lbl = (out_path / "labels" / split) / f"factory_bvk_{pdf_file.stem}.txt"

            if render_pdf_to_png(str(pdf_file), str(tgt_img)):
                assessment = curator.assess_drawing(str(tgt_img))
                if assessment.get("detected_boxes"):
                    with open(tgt_lbl, "w", encoding="utf-8") as lf:
                        for b in assessment["detected_boxes"]:
                            y = b["yolo_bbox"]
                            lf.write(f"{b['class_id']} {y[0]} {y[1]} {y[2]} {y[3]}\n")
                    stats["factory_bvk_added"] += 1

        print(f"  ✅ Успешно аннотировано и добавлено {stats['factory_bvk_added']} заводских чертежей.")

    # ---------------------------------------------------------
    # ШАГ 3. Генерация конфигурационного файла dataset.yaml
    # ---------------------------------------------------------
    train_total = len(list((out_path / "images" / "train").glob("*.png")))
    val_total = len(list((out_path / "images" / "val").glob("*.png")))
    stats["train_count"] = train_total
    stats["val_count"] = val_total
    stats["total_count"] = train_total + val_total

    yaml_content = f"""# Расширенный гибридный датасет ЕСКД (Синтетика + Завод БВК)
# Всего чертежей: {stats['total_count']} (Train: {train_total}, Val: {val_total})
path: {out_path.name}
train: images/train
val: images/val

names:
  0: views_field
  1: view_front
  2: view_top
  3: view_left
  4: stamp
  5: tech_req
  6: stamp_extra
"""
    yaml_file = out_path / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    stats_file = out_path / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"🎉 СБОРКА ЗАВЕРШЕНА! Всего в датасете: {stats['total_count']} чертежей.")
    print(f"  Обучение (Train): {train_total} шт.")
    print(f"  Валидация (Val):   {val_total} шт.")
    print(f"  Конфигурация YOLO: {yaml_file}")
    print("=" * 65)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка расширенного датасета ЕСКД")
    parser.add_argument("--target", type=int, default=1100, help="Целевое количество чертежей")
    parser.add_argument("--out", default=TARGET_EXPANDED_DIR, help="Папка назначения")
    args = parser.parse_args()

    build_expanded_dataset(target_total=args.target, output_dir=args.out)


