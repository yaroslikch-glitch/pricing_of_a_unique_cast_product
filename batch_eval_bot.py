# -*- coding: utf-8 -*-
"""
Автономный бот стресс-тестирования и оценки качества (Batch Eval & QA Bot).
Проект: Рекомендательный сервис оценки стоимости уникального литья (ЮУрГУ & ООО «БВК»).

Функции бота:
1. Автоматически сканирует директорию с чертежами (PDF, PNG, JPG).
2. «Скармливает» каждый чертеж в пайплайн веб-сервиса (детекция ЕСКД на AMD RX 6600 Vulkan + парсер ТТ + калькулятор).
3. Сохраняет структурированные данные (JSON, аннотированные изображения, Markdown отчет).
4. Анализирует аномалии (пропущенные штампы, нераспознанные марки сталей, странные массы/цены).
5. Формирует конкретный список рекомендаций по доработке кода.
"""

import os
import sys
import time
import json
import re
import argparse
from pathlib import Path
from PIL import Image

# Импорт логики веб-сервиса и калькулятора
import web_app
from cost_engine import FoundryCostEngine

class BatchEvalBot:
    def __init__(self, output_dir="bot_eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visuals_dir = self.output_dir / "annotated"
        self.visuals_dir.mkdir(parents=True, exist_ok=True)

    def extract_cost_number(self, cost_markdown):
        """Извлекает итоговую себестоимость в рублях из отчета markdown."""
        if not cost_markdown:
            return 0.0
        # Вариант 1: Ожидаемая стоимость (Expected) | **367 465 руб.**
        m1 = re.search(r"Ожидаемая стоимость[^\*]*\*\*\s*\|\s*\*\*([\d\s]+)\s*руб", cost_markdown, re.I)
        if m1:
            clean_str = m1.group(1).replace(" ", "").strip()
            try:
                return float(clean_str)
            except ValueError:
                pass
        
        # Вариант 2: ИТОГО себестоимость: 367 465 руб.
        m2 = re.search(r"(?:ИТОГО|Итого)\s*себестоимость[:\*\s]*([\d\s]+)\s*руб", cost_markdown, re.I)
        if m2:
            clean_str = m2.group(1).replace(" ", "").strip()
            try:
                return float(clean_str)
            except ValueError:
                pass
        return 0.0

    def evaluate_file(self, file_path, conf_threshold=0.35, has_pattern=False):
        """Прогон одного файла через пайплайн сервиса."""
        file_path = Path(file_path)
        t0 = time.time()
        
        # Вызов полного пайплайна веб-сервиса
        try:
            (
                annotated_img,
                views_crop,
                stamp_crop,
                status_info,
                mass_val,
                steel_val,
                group_val,
                heat_val,
                cost_summary_md,
                ops_md,
                analogs_md
            ) = web_app.analyze_drawing(str(file_path), conf_threshold=conf_threshold, has_pattern=has_pattern)
            
            elapsed_ms = (time.time() - t0) * 1000
            total_cost = self.extract_cost_number(cost_summary_md)
            
            # Извлекаем найденный аналог из analogs_md
            best_analog_match = re.search(r"Ближайший аналог в архиве БВК[:\*\s]+\*\*([^\*]+)\*\*", analogs_md)
            best_analog_name = best_analog_match.group(1).strip() if best_analog_match else "Не найден"

            # Сохраняем аннотированное изображение
            annotated_filename = f"eval_{file_path.stem}.png"
            if annotated_img is not None:
                annotated_img.save(self.visuals_dir / annotated_filename)

            # Анализ флагов и возможных проблем
            warnings = []
            if "не найдено элементов" in status_info:
                warnings.append("NO_ESKD_ELEMENTS")
            if stamp_crop is None:
                warnings.append("STAMP_MISSING")
            if views_crop is None and "NO_ESKD_ELEMENTS" not in warnings:
                warnings.append("VIEWS_MISSING")
            if total_cost == 0.0 and "NO_ESKD_ELEMENTS" not in warnings:
                warnings.append("ZERO_COST")
            if mass_val > 5000:
                warnings.append("EXTREME_MASS")

            res_record = {
                "file": file_path.name,
                "path": str(file_path),
                "status": "OK" if "NO_ESKD_ELEMENTS" not in warnings else "UNRECOGNIZED",
                "elapsed_ms": round(elapsed_ms, 1),
                "stamp_found": stamp_crop is not None,
                "views_found": views_crop is not None,
                "mass_kg": mass_val,
                "steel_grade": steel_val,
                "casting_group": group_val,
                "heat_treatment": heat_val,
                "best_analog": best_analog_name,
                "total_cost_rub": total_cost,
                "warnings": warnings,
                "annotated_preview": str(self.visuals_dir / annotated_filename)
            }
            return res_record

        except Exception as e:
            return {
                "file": file_path.name,
                "path": str(file_path),
                "status": "ERROR",
                "error": str(e),
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
                "warnings": ["CRASH_ERROR"]
            }

    def run_batch(self, files_list):
        """Пакетный запуск оценки по списку файлов."""
        print(f"\n=======================================================")
        print(f"🤖 [BATCH EVAL BOT] Старт автоматического тестирования")
        print(f"Количество тестовых чертежей: {len(files_list)}")
        print(f"=======================================================\n")

        results = []
        for i, f in enumerate(files_list, 1):
            print(f"[{i}/{len(files_list)}] Тестирование: {Path(f).name}...", end=" ", flush=True)
            rec = self.evaluate_file(f)
            status_symbol = "✅" if rec["status"] == "OK" else ("⚠️" if rec["status"] == "UNRECOGNIZED" else "❌")
            cost_str = f"{rec.get('total_cost_rub', 0):,.0f} руб.".replace(",", " ") if rec.get('total_cost_rub') else "N/A"
            print(f"{status_symbol} {rec['elapsed_ms']} мс | Сталь: {rec.get('steel_grade', '-')} | Масса: {rec.get('mass_kg', '-')} кг | {cost_str}")
            results.append(rec)

        # Сохранение полного JSON
        json_path = self.output_dir / "eval_results.json"
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[ГОТОВО] Структурированные данные сохранены: {json_path}")

        # Генерация итогового Markdown отчета с анализом
        report_path = self.generate_markdown_report(results)
        print(f"[ОТЧЕТ] Аналитический отчет сформирован: {report_path}")
        return results, report_path

    def generate_markdown_report(self, results):
        """Формирует структурированный отчет и рекомендации по доработке кода."""
        total = len(results)
        ok_count = sum(1 for r in results if r["status"] == "OK")
        unrec_count = sum(1 for r in results if r["status"] == "UNRECOGNIZED")
        err_count = sum(1 for r in results if r["status"] == "ERROR")
        avg_time = sum(r.get("elapsed_ms", 0) for r in results) / max(1, total)

        stamp_missing_count = sum(1 for r in results if "STAMP_MISSING" in r.get("warnings", []))
        no_views_count = sum(1 for r in results if "VIEWS_MISSING" in r.get("warnings", []))

        report_md = f"""# 🤖 Отчет Автоматического Тестирования и Стресс-Тестов (QA Bot)

> **Дата запуска:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
> **Протестировано чертежей:** {total}  
> **Среднее время на чертеж (Vulkan RX 6600):** {avg_time:.1f} мс  

---

## 📊 Сводные показатели качества

| Метрика | Значение | Доля |
| :--- | :---: | :---: |
| **Успешно распознано и оценено (OK)** | {ok_count} | {ok_count/total*100:.1f}% |
| **Чертежи без элементов ЕСКД (Unrecognized)** | {unrec_count} | {unrec_count/total*100:.1f}% |
| **Ошибки и сбои пайплайна (Errors)** | {err_count} | {err_count/total*100:.1f}% |
| **Отдельные виды (штамп отсутствует)** | {stamp_missing_count} | {stamp_missing_count/total*100:.1f}% |

---

## 🔍 Детализированная таблица результатов

| Имя файла | Статус | Время (мс) | Штамп | Виды | Марка стали | Масса (кг) | Аналог БВК | Итого (руб.) | Замечания |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
"""

        for r in results:
            stamp_s = "✅" if r.get("stamp_found") else "❌"
            views_s = "✅" if r.get("views_found") else "❌"
            cost_s = f"{r.get('total_cost_rub', 0):,.0f} руб.".replace(",", " ") if r.get("total_cost_rub") else "-"
            warn_s = ", ".join(r.get("warnings", [])) if r.get("warnings") else "—"
            analog_s = r.get("best_analog", "-")
            report_md += f"| `{r['file']}` | {r['status']} | {r['elapsed_ms']} | {stamp_s} | {views_s} | {r.get('steel_grade','-')} | {r.get('mass_kg','-')} | {analog_s} | {cost_s} | {warn_s} |\n"

        # Формирование рекомендаций по доработке кода
        recommendations = []
        if stamp_missing_count > 0:
            recommendations.append(
                f"- **Улучшение парсера штампа:** В {stamp_missing_count} случаях штамп не был обнаружен как отдельный bbox (чертеж содержит только вид или нестандартную рамку). Рекомендуется улучшить эвристику предсказания габаритной массы по площади проекции."
            )
        if unrec_count > 0:
            recommendations.append(
                f"- **Порог детекции:** {unrec_count} файлов не дали ни одной рамки при conf=0.35. Имеет смысл протестировать адаптивный порог conf=0.20 для слабоконтрастных сканов."
            )
        if any("EXTREME_MASS" in r.get("warnings", []) for r in results):
            recommendations.append(
                "- **Ограничитель массы:** Для некоторых видов расчетная масса превысила 5 тонн. Следует внедрить верхний порог отсечки для обычных деталей."
            )

        if not recommendations:
            recommendations.append("- Все протестированные чертежи обработаны без критических замечаний.")

        report_md += "\n---\n\n## 🛠️ Выявленные точки роста и Рекомендации по коду\n\n" + "\n".join(recommendations) + "\n"

        out_report_file = self.output_dir / "eval_report.md"
        out_report_file.write_text(report_md, encoding="utf-8")
        return out_report_file


def gather_sample_files(count=10):
    """Собирает репрезентативную выборку чертежей для тестирования бота."""
    samples = []
    
    # 1. Из валидационной выборки YOLO
    val_dir = Path("dataset_yolo/images/val")
    if val_dir.exists():
        val_files = list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg"))
        samples.extend(val_files[:count//2])

    # 2. Из извлеченных чертежей методичек
    mined_dir = Path("mined_dataset/images")
    if mined_dir.exists():
        mined_files = list(mined_dir.glob("*.png"))
        samples.extend(mined_files[:count//2])

    # 3. Из корня или тестов
    for p in Path(".").glob("*.png"):
        if not p.name.startswith("eval_") and p not in samples:
            samples.append(p)

    return samples[:count]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Автономный QA-бот оценки чертежей")
    parser.add_argument("--input-dir", type=str, default=None, help="Папка с чертежами для проверки")
    parser.add_argument("--count", type=int, default=12, help="Количество чертежей для теста")
    parser.add_argument("--out", type=str, default="bot_eval_results", help="Папка сохранения результатов")
    args = parser.parse_args()

    bot = BatchEvalBot(output_dir=args.out)

    if args.input_dir and Path(args.input_dir).exists():
        input_files = list(Path(args.input_dir).glob("*.png")) + list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.pdf"))
        target_files = input_files[:args.count]
    else:
        target_files = gather_sample_files(count=args.count)

    if not target_files:
        print("[ОШИБКА] Не найдено файлов чертежей для тестирования.")
        sys.exit(1)

    bot.run_batch(target_files)
