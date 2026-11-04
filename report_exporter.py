# -*- coding: utf-8 -*-
"""
Модуль экспорта калькуляций и коммерческих предложений (ТКП).
Формирует документы в форматах Excel (.xlsx) и PDF (.pdf) для металлургов и экономистов ООО «БВК» / ПГ КОНАР.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Регистрация кириллических шрифтов для ReportLab
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf"]:
    if os.path.exists(p):
        try:
            pdfmetrics.registerFont(TTFont("ArialCyr", p))
            FONT_REGULAR = "ArialCyr"
            break
        except Exception:
            pass

for p in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/tahomabd.ttf"]:
    if os.path.exists(p):
        try:
            pdfmetrics.registerFont(TTFont("ArialCyr-Bold", p))
            FONT_BOLD = "ArialCyr-Bold"
            break
        except Exception:
            pass


def _normalize_cost_data(cost_data: Any) -> Dict[str, Any]:
    """Приводит данные расчета к единому словарю независимо от формата (CostBreakdown или dict)."""
    if hasattr(cost_data, "total_cost_expected"):
        # Объект CostBreakdown из cost_engine.py
        cb = cost_data
        ndt = ["ВИК 100%"]
        if str(cb.casting_group) in ("2", "3", "3а"):
            ndt.append("МПД")
        if str(cb.casting_group) in ("3", "3а"):
            ndt.extend(["УЗК 100%", "Мех. испытания"])
        if str(cb.casting_group) == "3а":
            ndt.append("РК / Гидроиспытания")

        return {
            "tech_params": {
                "mass_kg": float(cb.mass_part_kg),
                "steel_grade": str(cb.steel_grade),
                "casting_group": str(cb.casting_group),
                "heat_treatment": getattr(cb, "heat_treatment", "нормализация_отпуск"),
                "ndt_methods": ndt,
                "complexity_factor": 1.15,
                "batch_size": 1
            },
            "cost_breakdown": {
                "metal_charge": float(cb.metal_cost),
                "melting_energy": float(cb.melting_energy_cost),
                "pattern_tooling": float(cb.tooling_pattern_cost),
                "molding_sand_cured": float(cb.molding_sand_cost),
                "heat_treatment": float(cb.heat_treatment_cost),
                "machining": float(cb.machining_rough_cost),
                "ndt_testing": float(cb.quality_control_cost),
                "overhead_expenses": float(cb.overhead_cost),
                "margin_profit": float(cb.scrap_risk_cost)
            },
            "total_cost_rub": float(cb.total_cost_expected)
        }
    return cost_data


def generate_excel_calculation(cost_data: Any,
                               recsys_data: Optional[Dict[str, Any]],
                               output_path: str = "calculation_report.xlsx") -> str:
    """Генерирует калькуляционную ведомость литого изделия в формате Microsoft Excel."""
    cost_data = _normalize_cost_data(cost_data)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Калькуляция отливки"
    ws.views.sheetView[0].showGridLines = True

    # Цветовая палитра БВК / Промышленный сланец
    fill_header = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid") # Deep Navy
    fill_sub = PatternFill(start_color="3B82F6", end_color="3B82F6", fill_type="solid")    # Cobalt Blue
    fill_accent = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid") # Light Slate
    fill_total = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")  # Neutral Total
    fill_highlight = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid") # Green highlight

    font_title = Font(name="Arial", size=14, bold=True, color="1E293B")
    font_header = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Arial", size=10, bold=True, color="1E293B")
    font_regular = Font(name="Arial", size=10, color="1E293B")

    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1")
    )
    thick_bottom = Border(bottom=Side(style="medium", color="1E3A8A"))

    # Шапка документа
    ws.merge_cells("A1:E1")
    ws["A1"] = "ООО «БВК» (ПРОМЫШЛЕННАЯ ГРУППА КОНАР) — ЛИТЕЙНЫЙ ДИВИЗИОН"
    ws["A1"].font = font_title
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")

    ws.merge_cells("A2:E2")
    ws["A2"] = f"Калькуляционная карта литого изделия • Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
    ws["A2"].font = Font(name="Arial", size=10, italic=True, color="64748B")

    # Секция 1: Параметры заготовки
    row = 4
    ws.merge_cells(f"A{row}:E{row}")
    ws[f"A{row}"] = "1. ПАРАМЕТРЫ ЛИТОЙ ЗАГОТОВКИ"
    ws[f"A{row}"].font = font_header
    ws[f"A{row}"].fill = fill_header

    tech = cost_data.get("tech_params", {})
    params = [
        ("Марка стали / сплава", tech.get("steel_grade", "20ГЛ")),
        ("Масса отливки (черновая)", f"{tech.get('mass_kg', 75.0):.1f} кг"),
        ("Масса расплава на заливку (КИМ = 0.55)", f"{tech.get('mass_kg', 75.0)/0.55:.1f} кг"),
        ("Группа отливки (ГОСТ 977-88)", str(tech.get("casting_group", "3"))),
        ("Вид термической обработки", tech.get("heat_treatment", "нормализация_отпуск")),
        ("Методы контроля (НК)", ", ".join(tech.get("ndt_methods", ["УЗК", "РК"]))),
        ("Технологическая сложность", f"Коэфф. {tech.get('complexity_factor', 1.0):.2f}"),
        ("Серийность заказа", f"{tech.get('batch_size', 1)} шт.")
    ]

    for label, val in params:
        row += 1
        ws[f"A{row}"] = label
        ws[f"A{row}"].font = font_bold
        ws[f"A{row}"].border = thin_border
        ws.merge_cells(f"B{row}:E{row}")
        ws[f"B{row}"] = val
        ws[f"B{row}"].font = font_regular
        ws[f"B{row}"].border = thin_border

    # Секция 2: Структура себестоимости
    row += 2
    ws.merge_cells(f"A{row}:E{row}")
    ws[f"A{row}"] = "2. КАЛЬКУЛЯЦИЯ ЗАТРАТ НА ИЗГОТОВЛЕНИЕ (РУБ.)"
    ws[f"A{row}"].font = font_header
    ws[f"A{row}"].fill = fill_header

    row += 1
    headers = ["№ п/п", "Статья затрат", "Норматив / Описание", "Доля, %", "Сумма, руб."]
    cols = ["A", "B", "C", "D", "E"]
    for c, h in zip(cols, headers):
        ws[f"{c}{row}"] = h
        ws[f"{c}{row}"].font = font_header
        ws[f"{c}{row}"].fill = fill_sub
        ws[f"{c}{row}"].alignment = Alignment(horizontal="center", vertical="center")
        ws[f"{c}{row}"].border = thin_border

    breakdown = cost_data.get("cost_breakdown", {})
    total_cost = float(cost_data.get("total_cost_rub", 1.0))

    cost_items = [
        ("1", "Металлошихта и ферросплавы", "Жидкий металл, шихта, угар", breakdown.get("metal_charge", 0)),
        ("2", "Модельный комплект (оснастка)", "Древесина/пластик, фрезеровка ЧПУ", breakdown.get("pattern_tooling", 0)),
        ("3", "Формовочные материалы (ХТС)", "Песок, смолы Альфасет, отвердители", breakdown.get("molding_sand_cured", 0)),
        ("4", "Плавка и заливка", "Электроэнергия ДСП, футеровка", breakdown.get("melting_energy", 0)),
        ("5", "Термическая обработка", "Газовые печи, нормализация/отпуск", breakdown.get("heat_treatment", 0)),
        ("6", "Финишная механическая обработка", "Обдирка, зачистка, карусельные станки", breakdown.get("machining", 0)),
        ("7", "Неразрушающий контроль (НК)", "УЗК, РК, магнитопорошковый контроль", breakdown.get("ndt_testing", 0)),
        ("8", "Цеховые и общезаводские расходы", "Амортизация, ФОТ, логистика", breakdown.get("overhead_expenses", 0)),
        ("9", "Плановая рентабельность", "Прибыль завода", breakdown.get("margin_profit", 0))
    ]

    for idx, name, desc, amount in cost_items:
        row += 1
        pct = (amount / total_cost) * 100 if total_cost > 0 else 0
        ws[f"A{row}"] = idx
        ws[f"B{row}"] = name
        ws[f"C{row}"] = desc
        ws[f"D{row}"] = round(pct, 1)
        ws[f"E{row}"] = round(amount)

        ws[f"A{row}"].alignment = Alignment(horizontal="center")
        ws[f"D{row}"].alignment = Alignment(horizontal="right")
        ws[f"E{row}"].alignment = Alignment(horizontal="right")
        ws[f"E{row}"].number_format = "#,##0 ₽"
        ws[f"D{row}"].number_format = "0.0%"

        for c in cols:
            ws[f"{c}{row}"].border = thin_border
            ws[f"{c}{row}"].font = font_regular

    # Итого
    row += 1
    ws[f"A{row}"] = ""
    ws[f"B{row}"] = "ИТОГО ОЦЕНКА СТОИМОСТИ (ОТПУСКНАЯ ЦЕНА)"
    ws[f"C{row}"] = ""
    ws[f"D{row}"] = 1.0
    ws[f"E{row}"] = round(total_cost)

    for c in cols:
        ws[f"{c}{row}"].font = font_bold
        ws[f"{c}{row}"].fill = fill_total
        ws[f"{c}{row}"].border = thin_border
    ws[f"E{row}"].number_format = "#,##0 ₽"
    ws[f"D{row}"].number_format = "100.0%"

    # Секция 3: Анализ прецедентов (RecSys CBR)
    if recsys_data and recsys_data.get("best_analog"):
        best = recsys_data["best_analog"]
        row += 2
        ws.merge_cells(f"A{row}:E{row}")
        ws[f"A{row}"] = "3. АНАЛИЗ ПРЕЦЕДЕНТОВ (RECSYS БВК & CASE-BASED REASONING)"
        ws[f"A{row}"].font = font_header
        ws[f"A{row}"].fill = fill_header

        recsys_rows = [
            ("Ближайшая деталь-аналог БВК", f"{best.get('id')} — {best.get('name')}"),
            ("Интегральное сходство (Multi-modal)", f"{best.get('sim_pct', 0)} %"),
            ("Визуальное сходство геометрии (DINOv2)", f"{best.get('sim_visual_pct', 0)} %"),
            ("Фактическая себестоимость аналога", f"{best.get('actual_cost_rub', 0):,} руб."),
            ("Статус модельной оснастки на складе", "ИМЕЕТСЯ В НАЛИЧИИ" if best.get("has_pattern") else "ОТСУТСТВУЕТ"),
            ("Потенциальная экономия на оснастке", f"{best.get('pattern_savings_rub', 0):,} руб."),
            ("Инженерная рекомендация", recsys_data.get("recommendation", ""))
        ]

        for label, val in recsys_rows:
            row += 1
            ws[f"A{row}"] = label
            ws[f"A{row}"].font = font_bold
            ws[f"A{row}"].border = thin_border
            ws.merge_cells(f"B{row}:E{row}")
            ws[f"B{row}"] = val
            ws[f"B{row}"].font = font_regular
            ws[f"B{row}"].border = thin_border
            if label == "Потенциальная экономия на оснастке" and best.get("pattern_savings_rub", 0) > 0:
                ws[f"B{row}"].fill = fill_highlight

    # Автоподбор ширины столбцов
    col_widths = {"A": 8, "B": 38, "C": 36, "D": 12, "E": 20}
    for c, w in col_widths.items():
        ws.column_dimensions[c].width = w

    wb.save(output_path)
    return str(Path(output_path).resolve())


def generate_pdf_tkp(cost_data: Any,
                     recsys_data: Optional[Dict[str, Any]],
                     output_path: str = "commercial_offer.pdf") -> str:
    """Генерирует официальное Технико-Коммерческое Предложение (ТКП) в формате PDF."""
    cost_data = _normalize_cost_data(cost_data)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    style_title = ParagraphStyle(
        "DocTitle",
        fontName=FONT_BOLD,
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=0,
        spaceAfter=4
    )
    style_subtitle = ParagraphStyle(
        "DocSubTitle",
        fontName=FONT_REGULAR,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12
    )
    style_section = ParagraphStyle(
        "DocSection",
        fontName=FONT_BOLD,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=6
    )
    style_cell = ParagraphStyle(
        "CellRegular",
        fontName=FONT_REGULAR,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )
    style_cell_bold = ParagraphStyle(
        "CellBold",
        fontName=FONT_BOLD,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )
    style_cell_right = ParagraphStyle(
        "CellRight",
        fontName=FONT_REGULAR,
        fontSize=8.5,
        leading=11,
        alignment=2,
        textColor=colors.HexColor("#1E293B")
    )
    style_cell_right_bold = ParagraphStyle(
        "CellRightBold",
        fontName=FONT_BOLD,
        fontSize=8.5,
        leading=11,
        alignment=2,
        textColor=colors.HexColor("#1E293B")
    )
    style_alert = ParagraphStyle(
        "AlertBox",
        fontName=FONT_REGULAR,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#065F46")
    )

    elements = []

    # Заголовок предприятия
    elements.append(Paragraph("ООО «БВК» • ПРОМЫШЛЕННАЯ ГРУППА КОНАР", style_title))
    elements.append(Paragraph(
        "Сталелитейное производство мирового уровня • г. Челябинск, ул. Енисейская, 8 • Тел.: +7 (351) 216-70-00<br/>"
        f"<b>ТЕХНИКО-КОММЕРЧЕСКОЕ ПРЕДЛОЖЕНИЕ (ТКП) № БВК-{datetime.now().strftime('%y%m%d-%H%M')}</b>",
        style_subtitle
    ))

    # 1. Параметры литой заготовки
    elements.append(Paragraph("1. Технологические параметры литой заготовки", style_section))
    tech = cost_data.get("tech_params", {})
    t1_data = [
        [
            Paragraph("<b>Марка сплава:</b>", style_cell_bold),
            Paragraph(tech.get("steel_grade", "20ГЛ"), style_cell),
            Paragraph("<b>Масса отливки:</b>", style_cell_bold),
            Paragraph(f"{tech.get('mass_kg', 75.0):.1f} кг", style_cell)
        ],
        [
            Paragraph("<b>Группа отливки:</b>", style_cell_bold),
            Paragraph(str(tech.get("casting_group", "3")) + " (ГОСТ 977-88)", style_cell),
            Paragraph("<b>Масса расплава:</b>", style_cell_bold),
            Paragraph(f"{tech.get('mass_kg', 75.0)/0.55:.1f} кг (КИМ 0.55)", style_cell)
        ],
        [
            Paragraph("<b>Термообработка:</b>", style_cell_bold),
            Paragraph(tech.get("heat_treatment", "нормализация_отпуск"), style_cell),
            Paragraph("<b>Методы контроля:</b>", style_cell_bold),
            Paragraph(", ".join(tech.get("ndt_methods", ["УЗК", "РК"])), style_cell)
        ]
    ]
    t1 = Table(t1_data, colWidths=[120, 140, 120, 140])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 8))

    # 2. Калькуляция затрат
    elements.append(Paragraph("2. Структура себестоимости и отпускная цена изделия", style_section))
    breakdown = cost_data.get("cost_breakdown", {})
    total_cost = float(cost_data.get("total_cost_rub", 1.0))

    t2_data = [
        [
            Paragraph("<b>№</b>", style_cell_bold),
            Paragraph("<b>Статья калькуляции</b>", style_cell_bold),
            Paragraph("<b>Нормативная база / Описание</b>", style_cell_bold),
            Paragraph("<b>Доля</b>", style_cell_right_bold),
            Paragraph("<b>Сумма, руб.</b>", style_cell_right_bold)
        ]
    ]

    items = [
        ("1", "Металлошихта и ферросплавы", "Жидкий металл, шихта, угар", breakdown.get("metal_charge", 0)),
        ("2", "Модельный комплект (оснастка)", "Изготовление/подготовка пресс-форм", breakdown.get("pattern_tooling", 0)),
        ("3", "Формовочные материалы (ХТС)", "Формовка Альфасет, кварцевый песок", breakdown.get("molding_sand_cured", 0)),
        ("4", "Плавка и разливка", "Электроэнергия ДСП-25, футеровка", breakdown.get("melting_energy", 0)),
        ("5", "Термическая обработка", "Нормализация с отпуском в газовых печах", breakdown.get("heat_treatment", 0)),
        ("6", "Финишная механическая обработка", "Обдирка литниковой системы, ЧПУ зачистка", breakdown.get("machining", 0)),
        ("7", "Неразрушающий контроль (НК)", "УЗК, РК, магнитопорошковая дефектоскопия", breakdown.get("ndt_testing", 0)),
        ("8", "Накладные и общецеховые расходы", "Общезаводской тариф, амортизация", breakdown.get("overhead_expenses", 0)),
        ("9", "Плановая прибыль завода", "Рентабельность инвестиций", breakdown.get("margin_profit", 0))
    ]

    for idx, name, desc, amt in items:
        pct = (amt / total_cost) * 100 if total_cost > 0 else 0
        t2_data.append([
            Paragraph(idx, style_cell),
            Paragraph(name, style_cell),
            Paragraph(desc, style_cell),
            Paragraph(f"{pct:.1f}%", style_cell_right),
            Paragraph(f"{amt:,.0f} ₽", style_cell_right)
        ])

    t2_data.append([
        Paragraph("", style_cell_bold),
        Paragraph("<b>ИТОГО ОТПУСКНАЯ ЦЕНА</b>", style_cell_bold),
        Paragraph("<b>С НДС и гарантией БВК</b>", style_cell_bold),
        Paragraph("<b>100.0%</b>", style_cell_right_bold),
        Paragraph(f"<b>{total_cost:,.0f} ₽</b>", style_cell_right_bold)
    ])

    t2 = Table(t2_data, colWidths=[20, 160, 180, 50, 110])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E2E8F0")),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 10))

    # 3. Анализ деталей-аналогов (CBR)
    if recsys_data and recsys_data.get("best_analog"):
        best = recsys_data["best_analog"]
        elements.append(Paragraph("3. Прецедентный анализ и рекомендации (RecSys & DINOv2 AI)", style_section))
        
        has_pat = best.get("has_pattern", False)
        pat_save = best.get("pattern_savings_rub", 0)

        t3_data = [
            [
                Paragraph("<b>Деталь-аналог БВК:</b>", style_cell_bold),
                Paragraph(f"{best.get('id')} — {best.get('name')}", style_cell),
                Paragraph("<b>Сходство геометрии (DINOv2):</b>", style_cell_bold),
                Paragraph(f"<b>{best.get('sim_visual_pct', 0)}%</b>", style_cell)
            ],
            [
                Paragraph("<b>Интегральный скор:</b>", style_cell_bold),
                Paragraph(f"{best.get('sim_pct', 0)}%", style_cell),
                Paragraph("<b>Наличие оснастки на складе:</b>", style_cell_bold),
                Paragraph("ДА (Готова к адаптации)" if has_pat else "НЕТ (Требуется изготовление)", style_cell)
            ]
        ]

        t3 = Table(t3_data, colWidths=[130, 150, 140, 100])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t3)
        elements.append(Spacer(1, 6))

        # Выделенный блок инженерной рекомендации
        rec_box_text = f"<b>Инженерное заключение:</b> {recsys_data.get('recommendation', '')}"
        if pat_save > 0 and has_pat:
            rec_box_text += f"<br/><b>Экономический эффект:</b> Снижение стоимости за счёт повторного применения модельного комплекта составит <b>до {pat_save:,.0f} руб.</b>, срок подготовки производства сокращается на 25 рабочих дней."
        
        rec_table = Table([[Paragraph(rec_box_text, style_alert)]], colWidths=[520])
        rec_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ECFDF5")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#10B981")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(rec_table)

    elements.append(Spacer(1, 14))
    # Блок подписей
    sig_data = [
        [
            Paragraph("<b>Главный металлург ООО «БВК»</b><br/><br/>__________________ / Смирнов А.В. /", style_cell),
            Paragraph("<b>Руководитель коммерческого управления</b><br/><br/>__________________ / Кузнецов Д.И. /", style_cell_right)
        ]
    ]
    t_sig = Table(sig_data, colWidths=[260, 260])
    t_sig.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(KeepTogether(t_sig))

    doc.build(elements)
    return str(Path(output_path).resolve())
