# -*- coding: utf-8 -*-
"""
Улучшенный модуль поиска и извлечения чертежей из инженерных методичек (Manual Dataset Miner).
Проект: Рекомендательный сервис оценки стоимости уникального литья (ЮУрГУ / ООО "БВК")

Ключевые обновления:
1. Строгий фильтр качества: отсекает страницы с текстом, параграфами, таблицами и водяными знаками библиотек.
2. Изоляция: сохраняет найденные чертежи ТОЛЬКО в staging-папку `mined_dataset/staging/`, не засоряя основной датасет.
3. Проверка пропорций и минимальной площади проекции чертежа (исключение мелких иллюстраций).
4. Режим `--clean`: очистка мусорных страниц.
"""

import os
import re
import sys
import json
import time
import shutil
import urllib.parse
import urllib.request
import ssl
from pathlib import Path
from PIL import Image

try:
    import pymupdf  # fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Стоп-слова: страницы учебников с таким содержанием отбрасываются
STOP_PAGE_PATTERNS = [
    r"репозиторий", r"библиотек", r"удк", r"ббк", r"isbn",
    r"содержание", r"введение", r"предисловие", r"оглавление",
    r"список\s+литературы", r"лабораторная\s+работа", r"практикум",
    r"методические\s+указания", r"титульный\s+лист", r"рецензент"
]

REGEX_STEEL = re.compile(r"\b(Сталь\s+\d+[А-Яа-я0-9]*Л?|ВЧ\s*\d+|СЧ\s*\d+|[0-9]{2}[А-Яа-я0-9]+Л)\b", re.I)
REGEX_NDT = re.compile(r"\b(УЗК|РК|МПД|ЦД|ВИК|радиографическ\w+|ультразвуков\w+|магнитопорошков\w+|капиллярн\w+)\b", re.I)
REGEX_HEAT = re.compile(r"\b(нормализац\w+|отжиг\w*|закалк\w+|отпуск\w*|аустенизац\w+|улучшени\w+)\b", re.I)

class ManualDatasetMiner:
    def __init__(self,
                 weights="eskd_yolo_best_ncnn_model",
                 output_dir="mined_dataset",
                 conf_thresh=0.45,
                 dpi=200):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.staging_images = self.output_dir / "staging_images"
        self.staging_labels = self.output_dir / "staging_labels"
        self.knowledge_dir = self.output_dir / "knowledge"

        for d in (self.staging_images, self.staging_labels, self.knowledge_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.dpi = dpi
        self.conf = conf_thresh

        # Загрузка модели детекции
        self.model = None
        if HAS_YOLO:
            if os.path.exists(weights):
                print(f"[Miner YOLO] Загрузка модели: {weights}")
                self.model = YOLO(weights)
            elif os.path.exists("eskd_yolo_best.pt"):
                print("[Miner YOLO] Загрузка PyTorch весов: eskd_yolo_best.pt")
                self.model = YOLO("eskd_yolo_best.pt")

    def is_valid_drawing_page(self, text: str, page) -> bool:
        """Проверяет, что страница является полноценным чертежом, а не страницей книги с текстом."""
        clean_text = text.strip()

        # 1. Если на странице более 350 символов связного текста — это книжная страница, а не чертеж
        if len(clean_text) > 350:
            return False

        # 2. Проверка стоп-слов (титулы, оглавления, репозитории)
        low_text = clean_text.lower()
        for pat in STOP_PAGE_PATTERNS:
            if re.search(pat, low_text):
                return False

        # 3. Чертеж должен содержать достаточное количество векторных линий
        drawings = page.get_drawings()
        if len(drawings) < 40:
            return False

        return True

    def process_pdf(self, pdf_path, max_pages=None):
        """Анализирует PDF-файл со строгим контролем чертежного содержания."""
        if not HAS_PYMUPDF:
            raise ImportError("Требуется PyMuPDF")

        pdf_path = Path(pdf_path)
        print(f"\n=======================================================")
        print(f"Анализ документа: {pdf_path.name}")
        print(f"=======================================================")

        doc = pymupdf.open(str(pdf_path))
        num_pages = min(len(doc), max_pages) if max_pages else len(doc)
        print(f"Страниц для аудита: {num_pages}")

        valid_drawings = 0
        knowledge_entries = []

        for pno in range(num_pages):
            page = doc[pno]
            text = page.get_text()

            # Сбор металлургических требований
            steels = list(set(REGEX_STEEL.findall(text)))
            ndt = list(set(REGEX_NDT.findall(text)))
            heat = list(set(REGEX_HEAT.findall(text)))

            if steels or ndt or heat:
                knowledge_entries.append({
                    "source_pdf": pdf_path.name,
                    "page": pno + 1,
                    "steels": steels,
                    "ndt_methods": ndt,
                    "heat_treatment": heat
                })

            # Фильтрация страницы: пропускаем текстовые страницы
            if not self.is_valid_drawing_page(text, page):
                continue

            # Растеризация в высоком разрешении для проверки YOLO
            mat = pymupdf.Matrix(self.dpi / 72.0, self.dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            page_stem = f"{pdf_path.stem}_p{pno+1:03d}"

            if self.model is not None:
                res = self.model.predict(img, conf=self.conf, verbose=False)[0]
                boxes = res.boxes

                # Требуем наличие поля проекций (views_field / view_front) с достаточной площадью
                has_large_view = False
                for b in boxes:
                    cls_id = int(b.cls.item())
                    xywhn = b.xywhn[0].tolist()
                    area = xywhn[2] * xywhn[3]
                    if cls_id in (0, 1) and area >= 0.12:
                        has_large_view = True
                        break

                if has_large_view:
                    img_filename = f"{page_stem}.png"
                    img.save(self.staging_images / img_filename)

                    yolo_lines = []
                    for b in boxes:
                        cls_id = int(b.cls.item())
                        xywhn = b.xywhn[0].tolist()
                        yolo_lines.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")

                    (self.staging_labels / f"{page_stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
                    valid_drawings += 1
                    print(f"  ✅ [Стр. {pno+1}] Отобран качественный чертеж -> {img_filename}")

        doc.close()

        if knowledge_entries:
            kb_path = self.knowledge_dir / f"{pdf_path.stem}_rules.json"
            kb_path.write_text(json.dumps(knowledge_entries, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[ИТОГ] Найдено подлинных чертежей: {valid_drawings} из {num_pages} страниц.")
        return valid_drawings

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Сбор и фильтрация чертежей из методичек")
    parser.add_argument("--dir", default="manuals_input", help="Директория с PDF-методичками")
    parser.add_argument("--clean", action="store_true", help="Очистить staging директорию")
    args = parser.parse_args()

    miner = ManualDatasetMiner()
    if args.clean:
        if miner.staging_images.exists():
            shutil.rmtree(miner.staging_images)
            miner.staging_images.mkdir(parents=True, exist_ok=True)
        if miner.staging_labels.exists():
            shutil.rmtree(miner.staging_labels)
            miner.staging_labels.mkdir(parents=True, exist_ok=True)
        print("[ОЧИСТКА] Папка staging очищена.")
        return

    pdf_files = list(Path(args.dir).glob("*.pdf"))
    if not pdf_files:
        print(f"В папке {args.dir} не найдено PDF файлов.")
        return

    print(f"Найдено PDF файлов для обработки: {len(pdf_files)}")
    for pdf_p in pdf_files:
        miner.process_pdf(pdf_p, max_pages=30)

if __name__ == "__main__":
    main()
