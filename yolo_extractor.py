# -*- coding: utf-8 -*-
"""
Модуль извлечения элементов чертежа через обученную модель YOLO (ООО "БВК" / ЮУрГУ)
Заменяет эвристический разбор листа на надежный детекционный пайплайн.
"""
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from ultralytics import YOLO

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "eskd_yolo_best.pt")

def load_drawing_image(file_path, dpi=200):
    """Загрузка чертежа из PDF или растрового файла (PNG/JPG)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        if not HAS_PYMUPDF:
            raise ImportError("Для чтения PDF требуется PyMuPDF (pip install pymupdf)")
        doc = fitz.open(file_path)
        page = doc[0]
        # Рендерим PDF в высоком разрешении
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    else:
        return Image.open(file_path).convert("RGB")

def prepare_dinov2_canvas(crop_pil, target_size=512):
    """
    Приводит вырезанное поле видов к квадрату 512x512 с пропорциональным
    масштабированием и центрированием на белом фоне (как при обучении DINOv2).
    """
    w, h = crop_pil.size
    side = int(max(w, h) * 1.05)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(crop_pil, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)

class ESKDExtractor:
    def __init__(self, weights_path=DEFAULT_WEIGHTS, conf=0.35, device="cpu"):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл весов YOLO не найден: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf = conf
        self.device = device

    def extract(self, input_source, out_dir="extracted_elements", save_crops=True):
        """
        Анализирует чертеж и извлекает:
          - views_field (для DINOv2)
          - stamp (для OCR / Qwen-VL)
          - tech_req (для анализа ТТ)
          - view_front, view_top, view_left
        """
        if isinstance(input_source, (str, Path)):
            img = load_drawing_image(str(input_source))
            base_name = os.path.splitext(os.path.basename(str(input_source)))[0]
        else:
            img = input_source
            base_name = "drawing"

        W, H = img.size
        results = self.model.predict(source=img, conf=self.conf, device=self.device, verbose=False)[0]

        extracted = {
            "image_size": (W, H),
            "views_field": None,
            "views_field_dinov2": None,
            "stamp": None,
            "tech_req": None,
            "stamp_extra": None,
            "views": {},
            "raw_boxes": []
        }

        # Словарь классов модели:
        # {0: 'views_field', 1: 'view_front', 2: 'view_top', 3: 'view_left', 4: 'stamp', 5: 'tech_req', 6: 'stamp_extra'}
        names = results.names

        # Находим лучшие боксы для каждого класса
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]

            # Защита от выхода за границы
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(x1 + 1, min(W, x2))
            y2 = max(y1 + 1, min(H, y2))

            crop = img.crop((x1, y1, x2, y2))
            extracted["raw_boxes"].append({
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": conf,
                "bbox": [x1, y1, x2, y2]
            })

            if cls_name == "views_field" and (extracted["views_field"] is None or conf > extracted["views_field"]["conf"]):
                extracted["views_field"] = {"crop": crop, "bbox": [x1, y1, x2, y2], "conf": conf}
                extracted["views_field_dinov2"] = prepare_dinov2_canvas(crop, 512)
            elif cls_name == "stamp" and (extracted["stamp"] is None or conf > extracted["stamp"]["conf"]):
                extracted["stamp"] = {"crop": crop, "bbox": [x1, y1, x2, y2], "conf": conf}
            elif cls_name == "tech_req" and (extracted["tech_req"] is None or conf > extracted["tech_req"]["conf"]):
                extracted["tech_req"] = {"crop": crop, "bbox": [x1, y1, x2, y2], "conf": conf}
            elif cls_name in ("view_front", "view_top", "view_left"):
                extracted["views"][cls_name] = {"crop": crop, "bbox": [x1, y1, x2, y2], "conf": conf}

        # Если views_field не найден как единый класс, объединяем отдельные виды
        if extracted["views_field"] is None and extracted["views"]:
            all_x1 = min(v["bbox"][0] for v in extracted["views"].values())
            all_y1 = min(v["bbox"][1] for v in extracted["views"].values())
            all_x2 = max(v["bbox"][2] for v in extracted["views"].values())
            all_y2 = max(v["bbox"][3] for v in extracted["views"].values())
            crop = img.crop((all_x1, all_y1, all_x2, all_y2))
            extracted["views_field"] = {"crop": crop, "bbox": [all_x1, all_y1, all_x2, all_y2], "conf": 0.8}
            extracted["views_field_dinov2"] = prepare_dinov2_canvas(crop, 512)

        # Сохранение результатов
        if save_crops and out_dir:
            out_folder = os.path.join(out_dir, base_name)
            os.makedirs(out_folder, exist_ok=True)

            # Сохраняем размеченное превью от YOLO
            annotated_arr = results.plot()
            annotated_img = Image.fromarray(annotated_arr)
            annotated_img.save(os.path.join(out_folder, "00_annotated.png"))

            if extracted["views_field"]:
                extracted["views_field"]["crop"].save(os.path.join(out_folder, "views_field_raw.png"))
                extracted["views_field_dinov2"].save(os.path.join(out_folder, "views_field_dinov2_512.png"))

            if extracted["stamp"]:
                extracted["stamp"]["crop"].save(os.path.join(out_folder, "stamp_title_block.png"))

            if extracted["tech_req"]:
                extracted["tech_req"]["crop"].save(os.path.join(out_folder, "tech_req.png"))

            for v_name, v_data in extracted["views"].items():
                v_data["crop"].save(os.path.join(out_folder, f"{v_name}.png"))

            print(f"[OK] Элементы чертежа сохранены в: {out_folder}")

        return extracted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Извлечение элементов чертежа ЕСКД через YOLO")
    parser.add_argument("--source", required=True, help="Путь к чертежу (PDF или PNG/JPG)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Путь к весам best.pt")
    parser.add_argument("--out", default="extracted_crops", help="Папка для сохранения вырезанных зон")
    parser.add_argument("--conf", type=float, default=0.35, help="Порог уверенности")
    args = parser.parse_args()

    extractor = ESKDExtractor(weights_path=args.weights, conf=args.conf)
    res = extractor.extract(args.source, out_dir=args.out)
    print("Найдено элементов:", len(res["raw_boxes"]))
