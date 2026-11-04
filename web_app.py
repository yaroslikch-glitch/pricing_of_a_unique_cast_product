# -*- coding: utf-8 -*-
"""
Интерактивный веб-сервис оценки стоимости уникального литого изделия.
Проект: ЮУрГУ & ООО «БВК» (ПГ КОНАР, г. Челябинск).
Стилизация: UI-UX-PRO-MAX (Industrial Enterprise Slate Theme).
Архитектура: RecSys Pipeline (рекомендательный поиск деталей-аналогов БВК с DINOv2 AI).
Экспорт: Официальное ТКП (PDF) и Калькуляционная ведомость (Excel .xlsx).
"""

import os
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import gradio as gr

from cost_engine import FoundryCostEngine, STEEL_PRICES, QUALITY_CONTROL_RATES, HEAT_TREATMENT_RATES
from tech_parser import parse_technical_text
from analog_recsys import RecsysAnalogMatcher
from report_exporter import generate_excel_calculation, generate_pdf_tkp

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Загрузка обученной модели (приоритет Vulkan NCNN)
MODEL_PATH = "eskd_yolo_best_ncnn_model" if os.path.exists("eskd_yolo_best_ncnn_model") else "eskd_yolo_best.pt"
print(f"[WEB APP] Инициализация модели: {MODEL_PATH}")
model = YOLO(MODEL_PATH) if (HAS_YOLO and os.path.exists(MODEL_PATH)) else None
cost_engine = FoundryCostEngine()
analog_matcher = RecsysAnalogMatcher()

# Цвета для классов разметки ЕСКД
CLASS_COLORS = {
    0: (0, 102, 204),    # views_field (синий)
    1: (0, 153, 76),     # view_front (зеленый)
    2: (204, 102, 0),    # view_top (оранжевый)
    3: (102, 0, 204),    # view_left (фиолетовый)
    4: (204, 0, 0),      # stamp (красный)
    5: (153, 0, 153),    # tech_req (пурпурный)
    6: (128, 128, 128)   # stamp_extra
}

CUSTOM_CSS = """
/* UI-UX-PRO-MAX: Industrial Enterprise Slate Theme */
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #e2e8f0 !important;
}
.header-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}
.badge-tech {
    display: inline-block;
    background: #1e3a8a;
    color: #93c5fd;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.badge-vulkan {
    display: inline-block;
    background: #701a75;
    color: #f472b6;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.badge-recsys {
    display: inline-block;
    background: #064e3b;
    color: #6ee7b7;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}
table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 12px 0 !important;
}
th {
    background-color: #1e293b !important;
    color: #cbd5e1 !important;
    padding: 10px 12px !important;
    font-size: 13px !important;
    border: 1px solid #334155 !important;
}
td {
    padding: 10px 12px !important;
    font-size: 13px !important;
    border: 1px solid #1e293b !important;
}
tr:hover {
    background-color: rgba(51, 65, 85, 0.5) !important;
}
"""

def load_input_image(file_obj, dpi=200):
    """Загрузка изображения или первой страницы PDF."""
    if file_obj is None:
        return None, ""
    
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    text_content = ""

    if file_path.lower().endswith(".pdf"):
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF не установлен.")
        doc = pymupdf.open(file_path)
        page = doc[0]
        text_content = page.get_text()
        mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img, text_content
    else:
        img = Image.open(file_path).convert("RGB")
        return img, text_content

def format_analogs_md(mass_kg, steel_grade, casting_group, calculated_cost, aspect_ratio=1.0, views_crop=None):
    """Формирует отчет рекомендательной системы аналогов БВК с DINOv2 сходством."""
    analogs = analog_matcher.find_analogs(
        mass_kg=mass_kg,
        steel_grade=steel_grade,
        casting_group=casting_group,
        aspect_ratio=aspect_ratio,
        views_crop=views_crop,
        k=3
    )
    effects = analog_matcher.apply_side_effects(analogs, calculated_cost)
    best = effects.get("best_analog")
    if not best:
        return "В базе прецедентов БВК пока не найдено близких аналогов."

    delta_sign = "+" if effects["cost_delta_rub"] >= 0 else ""
    delta_color = "#10b981" if abs(effects["cost_delta_pct"]) < 10 else "#f59e0b"

    md = f"""### 🎯 Рекомендательный поиск прецедентов (RecSys ООО «БВК» & DINOv2 AI)

> **Ближайший аналог в архиве БВК:** **{best['name']}** (`{best['id']}`)  
> **Интегральное сходство параметров:** **{best['sim_pct']}%** (Визуальное DINOv2: **{best.get('sim_visual_pct', 0)}%**)  
> **Фактическая заводская стоимость аналога:** **{best['actual_cost_rub']:,} руб.**  
> **Отклонение формульного расчета от факта:** <b style="color: {delta_color};">{delta_sign}{effects['cost_delta_rub']:,} руб. ({delta_sign}{effects['cost_delta_pct']}%)</b>  

---

#### 📋 Топ-3 исторических заказа БВК по степени конструктивной близости:

| Шифр детали | Наименование | Общее сходство | Геометрия (DINOv2) | Марка стали | Масса факт. | Себестоимость факт. | Модельная оснастка | Экономия на оснастке |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
""".replace(",", " ")

    for it in analogs:
        pattern_str = "Имеется на складе" if it.get("has_pattern") else "Требует изготовления"
        pat_savings = f"{it.get('pattern_savings_rub', 0):,} руб." if it.get("has_pattern") else "0 руб."
        md += f"| `{it['id']}` | **{it['name']}** | `{it['sim_pct']}%` | `{it.get('sim_visual_pct', 0)}%` | {it['steel_grade']} (гр. {it['casting_group']}) | {it['mass_kg']:.1f} кг | **{it['actual_cost_rub']:,} руб.** | {pattern_str} | <b style='color: #10b981;'>{pat_savings}</b> |\n".replace(",", " ")

    md += f"\n> 💡 **Инженерное заключение Case-Based Reasoning:** {effects.get('recommendation', '')}\n"
    return md

def compute_cost_display(mass_kg, steel_grade, casting_group, heat_treatment, has_pattern, stamp_missing=False, aspect_ratio=1.0, views_crop=None):
    """Формирует отчет калькулятора по параметрам и подбирает аналоги."""
    try:
        mass = float(mass_kg)
    except (ValueError, TypeError):
        mass = 50.0

    if mass <= 0:
        return "Укажите корректную массу детали (больше 0 кг).", "", ""

    calc = cost_engine.calculate(
        mass_kg=mass,
        steel_grade=str(steel_grade),
        casting_group=str(casting_group),
        heat_treatment=str(heat_treatment),
        has_existing_pattern=bool(has_pattern)
    )

    warning_note = ""
    if stamp_missing:
        warning_note = """
> ⚠️ **Внимание:** На чертеже не обнаружен штамп ГОСТ 2.104 с точной массой и маркой стали (загружен отдельный вид детали).  
> Масса оценена предварительно по площади проекции. Вы можете скорректировать её во вкладке **«Параметры технолога»** и нажать *«Пересчитать себестоимость»*.

---
"""

    summary_md = f"""
{warning_note}### 💰 Предварительная калькуляция себестоимости (ООО «БВК»)

| Показатель | Значение |
| :--- | :--- |
| **Ожидаемая стоимость (Expected)** | **{calc.total_cost_expected:,} руб.** |
| **Доверительный диапазон цен** | **{calc.total_cost_min:,} руб. — {calc.total_cost_max:,} руб.** |
| Масса готовой отливки | {calc.mass_part_kg} кг |
| Расчетная масса расплава (с прибылями) | {calc.mass_melt_kg} кг |
| Сталь / Группа отливки | {calc.steel_grade} / {calc.casting_group} |

---

#### 📊 Структура затрат по статьям:
* **Металл шихты и ферросплавы:** {calc.metal_cost:,} руб.
* **Энергия и плавка (ДСП/ИП):** {calc.melting_energy_cost:,} руб.
* **Модельная оснастка (Wood/Plastic ЧПУ):** {calc.tooling_pattern_cost:,} руб. {"*(учтена адаптация аналога)*" if has_pattern else "*(изготовление с нуля)*"}
* **Формовка ХТС (песчано-смоляные смеси):** {calc.molding_sand_cost:,} руб.
* **Термическая обработка:** {calc.heat_treatment_cost:,} руб.
* **Контроль качества (УЗК, РК, МПД, испытания):** {calc.quality_control_cost:,} руб.
* **Черновая механическая обработка:** {calc.machining_rough_cost:,} руб.
* **Технологический риск брака при контроле:** {calc.scrap_risk_cost:,} руб.
* **Общезаводские накладные расходы (18%):** {calc.overhead_cost:,} руб.
""".replace(",", " ")

    ops_md = "### 📋 Рекомендуемый технологический маршрут:\n\n"
    for i, op in enumerate(calc.operations_list, 1):
        ops_md += f"**{i}.** {op}\n\n"

    analogs_md = format_analogs_md(mass, str(steel_grade), str(casting_group), calc.total_cost_expected, aspect_ratio, views_crop=views_crop)

    return summary_md, ops_md, analogs_md

def analyze_drawing(file_obj, conf_threshold=0.35, has_pattern=False):
    """
    Полный сквозной конвейер анализа чертежа:
    1. Загрузка чертежа (PDF или изображение);
    2. Детекция зон через YOLOv8 на AMD RX 6600 (Vulkan NCNN);
    3. Парсинг параметров штампа и техтребований;
    4. Калькуляция себестоимости через FoundryCostEngine;
    5. Рекомендательный поиск деталей-аналогов через RecsysAnalogMatcher с DINOv2.
    """
    if file_obj is None:
        return None, None, None, "Загрузите файл чертежа (PDF или изображение).", 50.0, "20ГЛ", "3", "нормализация_отпуск", "", "", ""

    t0 = time.time()
    img, raw_pdf_text = load_input_image(file_obj)
    if img is None:
        return None, None, None, "Ошибка загрузки файла.", 50.0, "20ГЛ", "3", "нормализация_отпуск", "", "", ""

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    
    views_crop = None
    stamp_crop = None
    tech_crop = None
    detected_boxes = []
    aspect_ratio = 1.0

    if model is not None:
        res = model.predict(img, conf=conf_threshold, verbose=False)[0]
        boxes = res.boxes

        for b in boxes:
            cls_id = int(b.cls.item())
            cls_name = model.names.get(cls_id, str(cls_id))
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            color = CLASS_COLORS.get(cls_id, (0, 0, 0))

            # Отрисовка контура и метки
            draw.rectangle(xyxy, outline=color, width=4)
            draw.text((xyxy[0] + 5, xyxy[1] + 5), f"{cls_name} ({b.conf.item():.2f})", fill=color)

            # Вырезка ключевых элементов
            if cls_id == 0:  # views_field
                views_crop = img.crop(xyxy)
                vw, vh = views_crop.size
                aspect_ratio = vw / max(1, vh)
            elif cls_id in (1, 2, 3) and views_crop is None:  # view_front / top / left
                views_crop = img.crop(xyxy)
                vw, vh = views_crop.size
                aspect_ratio = vw / max(1, vh)
            elif cls_id == 4:  # stamp
                stamp_crop = img.crop(xyxy)
            elif cls_id == 5:  # tech_req
                tech_crop = img.crop(xyxy)

            detected_boxes.append(f"{cls_name} (conf={b.conf.item():.2f})")

    elapsed_ms = (time.time() - t0) * 1000

    # Если элементов чертежа не обнаружено вообще
    if len(detected_boxes) == 0:
        status_info = f"⚠️ На изображении не найдено элементов чертежа ЕСКД ({elapsed_ms:.1f} мс).\nОбнаружено элементов: 0."
        cost_summary_md = """### ⚠️ Чертеж не распознан
На загруженном изображении **не обнаружено проекционных видов или штампа ЕСКД**.
Система классифицировала изображение как посторонний объект (рукописная подпись, текст или фото без чертежа).
"""
        ops_md = "Маршрут не построен: чертежные проекции не обнаружены."
        analogs_md = "Поиск аналогов невозможен: геометрия не распознана."
        return (
            annotated,
            None,
            None,
            status_info,
            0.0,
            "20ГЛ",
            "3",
            "нормализация_отпуск",
            cost_summary_md,
            ops_md,
            analogs_md
        )

    # Инженерный парсинг реквизитов
    parsed_meta = parse_technical_text(raw_pdf_text)
    stamp_found = (stamp_crop is not None) or ("stamp" in raw_pdf_text.lower())
    
    # Интеллектуальный расчет массы детали
    if parsed_meta.get("mass_kg") is not None:
        mass_val = parsed_meta["mass_kg"]
    elif views_crop is not None:
        vw, vh = views_crop.size
        area_px = vw * vh
        estimated_mass = max(3.5, round((area_px / 1000000.0) * 55.0, 1))
        mass_val = min(estimated_mass, 950.0)
    else:
        mass_val = 35.0 if not stamp_found else 65.0

    steel_val = parsed_meta.get("steel_grade") or "20ГЛ"
    group_val = parsed_meta.get("casting_group") or "3"
    heat_val = parsed_meta.get("heat_treatment") or "нормализация_отпуск"

    # Расчет себестоимости и поиск аналогов с передачей views_crop для DINOv2
    cost_summary_md, ops_md, analogs_md = compute_cost_display(
        mass_val, steel_val, group_val, heat_val, has_pattern,
        stamp_missing=not stamp_found, aspect_ratio=aspect_ratio, views_crop=views_crop
    )

    status_info = f"⚡ Анализ завершен за {elapsed_ms:.1f} мс на AMD Radeon RX 6600 (Vulkan).\nОбнаружено элементов ЕСКД: {len(detected_boxes)}"
    if not stamp_found:
        status_info += " (Штамп не найден: масса рассчитана приблизительно по габаритам видов)."

    return (
        annotated,
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
    )

def export_pdf_action(mass_kg, steel_grade, casting_group, heat_treatment, has_pattern, views_crop=None):
    """Формирует и экспортирует официальный PDF документ ТКП."""
    os.makedirs("exports", exist_ok=True)
    out_file = f"exports/TKP_BVK_{int(time.time())}.pdf"
    calc = cost_engine.calculate(
        mass_kg=float(mass_kg),
        steel_grade=str(steel_grade),
        casting_group=str(casting_group),
        heat_treatment=str(heat_treatment),
        has_existing_pattern=bool(has_pattern)
    )
    analogs = analog_matcher.find_analogs(
        mass_kg=float(mass_kg),
        steel_grade=str(steel_grade),
        casting_group=str(casting_group),
        views_crop=views_crop,
        k=3
    )
    effects = analog_matcher.apply_side_effects(analogs, calc.total_cost_expected)
    path = generate_pdf_tkp(calc, effects, out_file)
    return path

def export_excel_action(mass_kg, steel_grade, casting_group, heat_treatment, has_pattern, views_crop=None):
    """Формирует и экспортирует калькуляционную ведомость в Excel."""
    os.makedirs("exports", exist_ok=True)
    out_file = f"exports/Calc_BVK_{int(time.time())}.xlsx"
    calc = cost_engine.calculate(
        mass_kg=float(mass_kg),
        steel_grade=str(steel_grade),
        casting_group=str(casting_group),
        heat_treatment=str(heat_treatment),
        has_existing_pattern=bool(has_pattern)
    )
    analogs = analog_matcher.find_analogs(
        mass_kg=float(mass_kg),
        steel_grade=str(steel_grade),
        casting_group=str(casting_group),
        views_crop=views_crop,
        k=3
    )
    effects = analog_matcher.apply_side_effects(analogs, calc.total_cost_expected)
    path = generate_excel_calculation(calc, effects, out_file)
    return path

# Интерфейс Gradio с темой Industrial Enterprise Slate
with gr.Blocks(title="Ценообразование отливок — ЮУрГУ & БВК") as demo:
    gr.HTML("""
    <div class="header-box">
        <div style="font-size: 22px; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">🏛️ Интеллектуальный сервис ценообразования уникального литого изделия</div>
        <div style="font-size: 13px; color: #94a3b8; line-height: 1.6;">
            Индустриальный партнёр: <b>ООО «БВК» (ПГ КОНАР, г. Челябинск)</b> | Учебный трек <b>ЮУрГУ</b><br>
            <span class="badge-tech">ГОСТ 977-88</span>
            <span class="badge-vulkan">Vulkan NCNN (AMD RX 6600)</span>
            <span class="badge-recsys">RecSys DINOv2 AI</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            input_file = gr.File(label="📄 Загрузите чертеж изделия (PDF / PNG / JPG)", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
            conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.35, step=0.05, label="Порог уверенности детектора (YOLO Conf)")
            has_pattern_cb = gr.Checkbox(label="Есть близкий аналог модельной оснастки (скидка на модель)", value=False)
            btn_analyze = gr.Button("🔍 Проанализировать чертеж и подобрать аналоги", variant="primary")
            status_text = gr.Textbox(label="Статус системы", interactive=False)

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.TabItem("📐 Разметка ЕСКД"):
                    out_annotated = gr.Image(label="Разметка элементов чертежа (ГОСТ 2.104)")
                    with gr.Row():
                        out_views = gr.Image(label="Поле видов детали")
                        out_stamp = gr.Image(label="Штамп чертежа")

                with gr.TabItem("🔍 Детали-аналоги (RecSys БВК)"):
                    out_analogs = gr.Markdown("Загрузите чертеж для подбора близких прецедентов из заводского архива.")

                with gr.TabItem("💵 Калькуляция себестоимости"):
                    out_cost_summary = gr.Markdown("Загрузите чертеж для расчета сметы.")

                with gr.TabItem("⚙️ Параметры технолога"):
                    gr.Markdown("Параметры, извлеченные из штампа и техтребований (можно скорректировать для перерасчета):")
                    with gr.Row():
                        in_mass = gr.Number(label="Масса детали (кг)", value=75.0)
                        in_steel = gr.Dropdown(choices=list(STEEL_PRICES.keys())[:-1], label="Марка стали / сплава", value="20ГЛ")
                    with gr.Row():
                        in_group = gr.Dropdown(choices=["1", "2", "3", "3а"], label="Группа отливки (ГОСТ 977-88)", value="3")
                        in_heat = gr.Dropdown(choices=["нормализация", "нормализация_отпуск", "закалка_отпуск", "аустенизация", "отжиг"], label="Термообработка", value="нормализация_отпуск")
                    btn_recalc = gr.Button("🔄 Пересчитать себестоимость и аналоги", variant="secondary")

                with gr.TabItem("🛠️ Технологический маршрут"):
                    out_ops = gr.Markdown("Список операций будет сформирован после анализа.")

                with gr.TabItem("📄 Экспорт документов (PDF / Excel)"):
                    gr.Markdown("### 📥 Скачать официальную документацию (ПГ КОНАР / ООО «БВК»)")
                    gr.Markdown("Формирование коммерческого предложения (ТКП) с печатью и детальной калькуляционной карты затрат для передачи заказчику.")
                    with gr.Row():
                        btn_export_pdf = gr.Button("📄 Скачать официальное ТКП (PDF)", variant="primary")
                        btn_export_excel = gr.Button("📊 Скачать калькуляционную ведомость (Excel)", variant="secondary")
                    with gr.Row():
                        out_pdf_file = gr.File(label="Готовое ТКП (PDF)", interactive=False)
                        out_excel_file = gr.File(label="Калькуляционная карта (Excel .xlsx)", interactive=False)

    # Логика кнопок анализа и пересчета
    btn_analyze.click(
        fn=analyze_drawing,
        inputs=[input_file, conf_slider, has_pattern_cb],
        outputs=[
            out_annotated,
            out_views,
            out_stamp,
            status_text,
            in_mass,
            in_steel,
            in_group,
            in_heat,
            out_cost_summary,
            out_ops,
            out_analogs
        ]
    )

    btn_recalc.click(
        fn=compute_cost_display,
        inputs=[in_mass, in_steel, in_group, in_heat, has_pattern_cb, gr.State(False), gr.State(1.0), out_views],
        outputs=[out_cost_summary, out_ops, out_analogs]
    )

    # Логика экспорта документов
    btn_export_pdf.click(
        fn=export_pdf_action,
        inputs=[in_mass, in_steel, in_group, in_heat, has_pattern_cb, out_views],
        outputs=[out_pdf_file]
    )

    btn_export_excel.click(
        fn=export_excel_action,
        inputs=[in_mass, in_steel, in_group, in_heat, has_pattern_cb, out_views],
        outputs=[out_excel_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS)
