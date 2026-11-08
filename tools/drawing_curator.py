# -*- coding: utf-8 -*-
"""
Интеллектуальный куратор и фильтратор датасета чертежей (Drawing Curator).
Используется субагентом `dataset-miner`.
Оценивает Suitability Score (0-100%) чертежей через двухуровневую нейро-воронку:
1. OpenCV (DPI, чёткость, размытие, заполненность).
2. Обученная YOLOv8 NCNN (наличие штампа ЕСКД, виды, разрезы, площадь проекций).
"""

import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from ultralytics import YOLO

DEFAULT_MODEL_PATH = "eskd_yolo_best.pt"
FALLBACK_PT_MODEL = "eskd_yolo_best.pt"

class DrawingCurator:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        if not os.path.exists(self.model_path) and os.path.exists(FALLBACK_PT_MODEL):
            self.model_path = FALLBACK_PT_MODEL
        print(f"[CURATOR] Загрузка модели оценки чертежей: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")

    def render_pdf_first_page(self, pdf_path: str, dpi: int = 150) -> np.ndarray:
        """Рендерит первую страницу PDF в изображение OpenCV BGR."""
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            raise ValueError(f"Пустой PDF: {pdf_path}")
        page = doc[0]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        doc.close()
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def assess_drawing(self, file_path: str) -> Dict[str, Any]:
        """
        Оценивает чертеж и вычисляет Suitability Score (0-100).
        """
        path = Path(file_path)
        if not path.exists():
            return {"file": str(path), "suitability_score": 0, "status": "ERROR", "reason": "File not found"}

        # 1. Загрузка в изображение
        try:
            if path.suffix.lower() == ".pdf":
                img = self.render_pdf_first_page(str(path))
            else:
                img = cv2.imread(str(path))
                if img is None:
                    return {"file": path.name, "suitability_score": 0, "status": "ERROR", "reason": "Cannot read image"}
        except Exception as e:
            return {"file": path.name, "suitability_score": 0, "status": "ERROR", "reason": f"Corrupt file: {e}"}

        h, w = img.shape[:2]
        total_pixels = h * w

        # 2. Оценка качества графики (OpenCV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_sharp = laplacian_var > 80.0
        is_hd = (min(h, w) >= 1000)

        # 3. Инференс обученной модели YOLOv8
        results = self.model(img, conf=0.25, verbose=False)
        boxes = results[0].boxes

        stamp_detected = False
        tech_req_detected = False
        views_count = 0
        total_views_area = 0.0

        detected_boxes = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            bw = xyxy[2] - xyxy[0]
            bh = xyxy[3] - xyxy[1]
            box_area = bw * bh

            # Классы модели: 0: views_field, 1: view_front, 2: view_top, 3: view_left, 4: stamp, 5: tech_req, 6: stamp_extra
            if cls_id in [4, 6]:  # stamp / stamp_extra
                stamp_detected = True
            elif cls_id in [0, 1, 2, 3]:  # views_field, view_front, view_top, view_left
                views_count += 1
                total_views_area += box_area
            elif cls_id == 5:  # tech_req
                tech_req_detected = True

            detected_boxes.append({
                "class_id": cls_id,
                "conf": round(conf, 3),
                "xyxy": [round(x, 1) for x in xyxy],
                "yolo_bbox": [
                    round((xyxy[0] + bw / 2) / w, 4),
                    round((xyxy[1] + bh / 2) / h, 4),
                    round(bw / w, 4),
                    round(bh / h, 4)
                ]
            })

        views_coverage_pct = round((total_views_area / total_pixels) * 100.0, 1)

        # 4. Расчет индекса пригодности (Suitability Score)
        # Критерии:
        # - Наличие штампа: 30 баллов
        # - Наличие проекций детали: 40 баллов (зависит от числа и площади)
        # - Качество изображения (HD + чёткость): 15 баллов
        # - Полнота заполнения листа (без пустот и без перегруза текстом): 15 баллов
        score = 0.0

        # Штамп (30 баллов)
        if stamp_detected:
            score += 30.0

        # Виды и проекции (40 баллов)
        if views_count >= 1:
            score += 15.0
            if views_count >= 2:
                score += 15.0  # минимум 2 проекции (вид спереди + разрез/сбоку)
            if 15.0 <= views_coverage_pct <= 75.0:
                score += 10.0  # проекции занимают адекватную площадь листа

        # Качество графики (15 баллов)
        if is_hd:
            score += 8.0
        if is_sharp:
            score += 7.0

        # Пропорции и компоновка (15 баллов)
        aspect = max(w / h, h / w)
        if 1.2 <= aspect <= 1.6:  # стандартные форматы А4, А3, А2 по ГОСТ 2.301
            score += 10.0
        if gray.mean() > 180:  # белый фон технического чертежа
            score += 5.0

        if tech_req_detected: score += 5.0
        suitability_score = round(min(100.0, score), 1)

        # Категоризация
        if suitability_score >= 80.0:
            category = "GOLDEN"
            verdict = "Идеальный деталировочный чертеж ЕСКД. Подходит для дообучения и оценки."
        elif suitability_score >= 60.0:
            category = "SILVER"
            verdict = "Годный чертеж, но требуется контроль (частично смазан штамп или один вид)."
        else:
            category = "REJECTED"
            reasons = []
            if not stamp_detected:
                reasons.append("нет штампа ЕСКД")
            if views_count == 0:
                reasons.append("нет видов детали (схема/текст)")
            if not is_hd:
                reasons.append("низкое разрешение")
            verdict = f"Отсеян: {', '.join(reasons)}."

        return {
            "filename": path.name,
            "path": str(path),
            "suitability_score": suitability_score,
            "category": category,
            "verdict": verdict,
            "resolution": f"{w}x{h}",
            "sharpness_var": round(laplacian_var, 1),
            "stamp_detected": stamp_detected,
            "views_count": views_count,
            "views_coverage_pct": views_coverage_pct,
            "detected_boxes": detected_boxes
        }

    def curate_directory(self, input_dir: str, output_base: str = "curated_drawings") -> Dict[str, Any]:
        """
        Сканирует папку с сырыми чертежами, сортирует их по категориям и генерирует отчет.
        """
        inp = Path(input_dir)
        out = Path(output_base)
        golden_dir = out / "golden"
        silver_dir = out / "silver"
        rejected_dir = out / "rejected"

        golden_dir.mkdir(parents=True, exist_ok=True)
        silver_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in inp.iterdir() if p.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg"]]
        results = []

        print(f"\n[CURATOR] Старт нейро-фильтрации {len(files)} чертежей из: {input_dir}")
        for i, f in enumerate(files, 1):
            assessment = self.assess_drawing(str(f))
            cat = assessment["category"]
            score = assessment["suitability_score"]
            print(f"  [{i:02d}/{len(files):02d}] {f.name[:32]:<32} -> {score:5.1f}% [{cat}]")

            # Копирование в соответствующую папку
            target_dir = golden_dir if cat == "GOLDEN" else (silver_dir if cat == "SILVER" else rejected_dir)
            target_file = target_dir / f.name
            shutil.copy2(str(f), str(target_file))

            # Если чертеж GOLDEN — создаем готовый YOLO .txt файл аннотаций
            if cat == "GOLDEN" and assessment.get("detected_boxes"):
                txt_name = target_file.with_suffix(".txt")
                with open(txt_name, "w", encoding="utf-8") as tf:
                    for b in assessment["detected_boxes"]:
                        yolo = b["yolo_bbox"]
                        tf.write(f"{b['class_id']} {yolo[0]} {yolo[1]} {yolo[2]} {yolo[3]}\n")

            results.append(assessment)

        summary = {
            "total_processed": len(results),
            "golden_count": sum(1 for r in results if r["category"] == "GOLDEN"),
            "silver_count": sum(1 for r in results if r["category"] == "SILVER"),
            "rejected_count": sum(1 for r in results if r["category"] == "REJECTED"),
            "results": results
        }

        # Сохраняем отчет в JSON
        report_file = out / "curation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n[CURATOR] Завершено! Золотых: {summary['golden_count']}, Требуют проверки: {summary['silver_count']}, Отсеяно: {summary['rejected_count']}")
        print(f"[CURATOR] Отчет сохранен: {report_file}")
        return summary

if __name__ == "__main__":
    curator = DrawingCurator()
    # Тест на 5 чертежах из заводского архива
    test_dir = r"C:\Users\Yaroslav\OneDrive\Desktop\датасет проекта\датабаза\drawings"
    if os.path.exists(test_dir):
        # Создаем временную тестовую выборку для демонстрации
        temp_sample = Path("temp_curator_sample")
        temp_sample.mkdir(exist_ok=True)
        sample_files = list(Path(test_dir).glob("*.pdf"))[:5]
        for s in sample_files:
            shutil.copy2(str(s), str(temp_sample / s.name))

        curator.curate_directory(str(temp_sample), "curated_drawings_demo")
        shutil.rmtree(temp_sample)


