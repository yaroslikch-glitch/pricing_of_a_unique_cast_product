# -*- coding: utf-8 -*-
"""
Инструмент металлургического и технологического аудита литой детали.
Используется субагентом «Технолог-Литейщик» (casting-technologist).
Стандарты: ГОСТ 977-88, ГОСТ 26645-85, ГОСТ 3212-92.
"""

import sys
import os
import json
import math
import urllib.request
from typing import Dict, Any, Optional

REMOTE_DEEPSEEK_URL = os.environ.get("DEEPSEEK_API_URL", "http://78.17.76.203:9655/v1/chat/completions")

# Физико-механические и литейные свойства марок сталей по ГОСТ 977-88
STEEL_FOUNDRY_PROPERTIES = {
    "20Л": {
        "shrinkage_linear_pct": 1.9,
        "liquidus_temp_c": 1520,
        "pouring_temp_c": 1560,
        "crack_sensitivity": "Низкая",
        "deoxidation": "Al + SiCa (дробное раскисление)",
        "heat_treatment": "Нормализация 880-900°C или отжиг"
    },
    "25Л": {
        "shrinkage_linear_pct": 2.0,
        "liquidus_temp_c": 1515,
        "pouring_temp_c": 1555,
        "crack_sensitivity": "Средняя",
        "deoxidation": "Al + FeSi + SiCa",
        "heat_treatment": "Нормализация 870-890°C + высокий отпуск 620-650°C"
    },
    "20ГЛ": {
        "shrinkage_linear_pct": 2.05,
        "liquidus_temp_c": 1510,
        "pouring_temp_c": 1550,
        "crack_sensitivity": "Повышенная (хладноломкость, дендритная ликвация)",
        "deoxidation": "Комплексное (Al + SiCa + Ti модифицирование)",
        "heat_treatment": "Нормализация 890-910°C + отпуск 600-630°C (KCV-60°C >= 25 Дж/см2)"
    },
    "20ХМЛ": {
        "shrinkage_linear_pct": 2.1,
        "liquidus_temp_c": 1500,
        "pouring_temp_c": 1545,
        "crack_sensitivity": "Высокая (склонность к флокенам и горячим трещинам)",
        "deoxidation": "Вакуумирование + SiCa",
        "heat_treatment": "Закалка в масло 900-920°C + высокий отпуск 660-680°C"
    },
    "15Х1М1ФЛ": {
        "shrinkage_linear_pct": 2.2,
        "liquidus_temp_c": 1495,
        "pouring_temp_c": 1540,
        "crack_sensitivity": "Очень высокая (жаропрочная перлитная сталь)",
        "deoxidation": "Глубокое вакуумирование в ковше",
        "heat_treatment": "Нормализация 980-1000°C + двухступенчатый отпуск 710-740°C"
    },
    "08Х18Н10ТЛ": {
        "shrinkage_linear_pct": 2.4,
        "liquidus_temp_c": 1440,
        "pouring_temp_c": 1510,
        "crack_sensitivity": "Крайне высокая (высокий КТР, риск межкристаллитной коррозии)",
        "deoxidation": "Безуглеродистый феррохром, продувка аргоном",
        "heat_treatment": "Аустенизация 1050-1080°C с закалкой в воду"
    }
}

def analyze_casting_tech(part_name: str,
                         steel_grade: str,
                         mass_kg: float,
                         dimensions_mm: list = None,
                         casting_group: str = "3",
                         with_ai_reasoning: bool = True) -> Dict[str, Any]:
    """
    Выполняет комплексный инженерный расчет технологичности отливки.
    """
    if dimensions_mm is None or len(dimensions_mm) < 3:
        dimensions_mm = [300, 300, 150]

    norm_steel = "20ГЛ"
    for s in STEEL_FOUNDRY_PROPERTIES:
        if s in steel_grade.upper():
            norm_steel = s
            break

    props = STEEL_FOUNDRY_PROPERTIES.get(norm_steel, STEEL_FOUNDRY_PROPERTIES["20ГЛ"])

    # 1. Расчет металлургического баланса
    # Для ответственных деталей группы 3 КИМ = 0.52 - 0.58
    yield_ratio = 0.55 if casting_group in ["3", "3а"] else 0.62
    mass_melt_kg = round(mass_kg / yield_ratio, 1)
    riser_mass_kg = round(mass_melt_kg - mass_kg, 1)

    # Объемное сжатие при остывании
    vol_shrinkage_pct = round(props["shrinkage_linear_pct"] * 3.0, 1)
    void_volume_dm3 = round((mass_kg / 7.85) * (vol_shrinkage_pct / 100.0), 2)

    # 2. Формовочная смесь ХТС
    sand_mass_ton = round((mass_melt_kg * 4.2) / 1000.0, 2)
    binder_furane_kg = round(sand_mass_ton * 1000.0 * 0.012, 1) # 1.2% смолы

    # 3. Энергетика плавки
    kwh_dsp25 = round(mass_melt_kg * 0.65 * 1.35)

    # 4. Программа неразрушающего контроля (НК) по ГОСТ 977-88
    ndt_program = [
        "100% Визуально-измерительный контроль (ВИК) по РД 03-606-03",
        r"Механические испытания на растяжение (\sigma_b, \sigma_T, \delta, \psi) и ударную вязкость KCV при -40°C",
    ]
    if casting_group in ["2", "3", "3а"]:
        ndt_program.append("Магнитопорошковая дефектоскопия (МПД) сопряжений и галтелей по ГОСТ 21105-87")
    if casting_group in ["3", "3а"]:
        ndt_program.append("100% Ультразвуковой контроль (УЗК) литых сопряжений и тепловых узлов по ГОСТ 24507-80 (группа качества 2)")
    if casting_group == "3а":
        ndt_program.append("Радиографический контроль (РК) зон сопряжений и фланцев по ГОСТ 23055-78")
        ndt_program.append("Гидравлические испытания на плотность металла Pпр = 1.5 Pраб")

    tech_summary = {
        "part_name": part_name,
        "steel_grade": norm_steel,
        "casting_group": f"Группа {casting_group} ГОСТ 977-88",
        "mass_part_kg": mass_kg,
        "mass_melt_kg": mass_melt_kg,
        "riser_mass_kg": riser_mass_kg,
        "yield_ratio_kim": yield_ratio,
        "void_shrinkage_volume_dm3": void_volume_dm3,
        "linear_shrinkage_pct": props["shrinkage_linear_pct"],
        "pouring_temp_c": props["pouring_temp_c"],
        "sand_mass_ton": sand_mass_ton,
        "binder_furane_kg": binder_furane_kg,
        "melting_energy_kwh": kwh_dsp25,
        "crack_sensitivity": props["crack_sensitivity"],
        "deoxidation_method": props["deoxidation"],
        "heat_treatment_norm": props["heat_treatment"],
        "ndt_program": ndt_program
    }

    ai_conclusion = ""
    ai_reasoning = ""

    if with_ai_reasoning:
        try:
            prompt = f"""
Ты — Главный технолог литейного производства завода БВК/КОНАР.
Дай экспертное технологическое заключение по отливке:
- Деталь: {part_name}
- Марка стали: {norm_steel} (Линейная усадка {props['shrinkage_linear_pct']}%, интервал заливки ~{props['pouring_temp_c']}°C)
- Чистая масса: {mass_kg} кг | Масса расплава заливки: {mass_melt_kg} кг (КИМ {yield_ratio})
- Габариты: {dimensions_mm[0]}x{dimensions_mm[1]}x{dimensions_mm[2]} мм
- Группа отливки: {casting_group} ГОСТ 977-88

СФОРМУЛИРУЙ В ЗАКЛЮЧЕНИИ:
1. Оценка технологичности геометрии (балл 1-10) и риски образования усадочных раковин в тепловых узлах.
2. Рекомендации по литниково-питающей системе (размещение прибылей, экзотермических вставок или наружных холодильников).
3. Предупреждения по трещинообразованию для марки {norm_steel} при кристаллизации в ХТС-форме.
4. Контрольные точки для отдела главного технолога и ОТК.
Отвечай профессиональным языком заводского литейщика (лаконично, строго по делу).
"""
            req_data = json.dumps({
                "model": "deepseek-r1",
                "messages": [
                    {"role": "system", "content": "You are the Chief Foundry Technologist at a heavy metallurgical plant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 4096
            }).encode("utf-8")

            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(
                REMOTE_DEEPSEEK_URL,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with opener.open(req, timeout=90) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                choice = res["choices"][0]
                ai_conclusion = choice["message"].get("content", "")
                ai_reasoning = choice["message"].get("reasoning_content", "")
        except Exception as e:
            ai_conclusion = f"Аудит DeepSeek-R1 временно недоступен: {e}"

    tech_summary["ai_conclusion"] = ai_conclusion
    tech_summary["ai_reasoning"] = ai_reasoning
    return tech_summary

if __name__ == "__main__":
    res = analyze_casting_tech(
        part_name="Вал ступенчатый редуктора",
        steel_grade="20ГЛ",
        mass_kg=100.8,
        dimensions_mm=[1200, 220, 220],
        casting_group="3"
    )
    print("=" * 60)
    print(f"ТЕХНОЛОГИЧЕСКИЙ АУДИТ: {res['part_name']} ({res['steel_grade']})")
    print(f"Масса отливки: {res['mass_part_kg']} кг | Металл заливки: {res['mass_melt_kg']} кг | КИМ: {res['yield_ratio_kim']}")
    print(f"Объем усадочной раковины без прибылей: {res['void_shrinkage_volume_dm3']} дм³")
    print(f"Программа НК: {len(res['ndt_program'])} операций")
    print("=" * 60)
    print("ЗАКЛЮЧЕНИЕ ГЛАВНОГО ТЕХНОЛОГА-ЛИТЕЙЩИКА:")
    print(res["ai_conclusion"])

