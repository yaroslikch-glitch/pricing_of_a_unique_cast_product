# -*- coding: utf-8 -*-
"""
Металлургический калькулятор себестоимости уникального литого изделия.
Разработан с учетом нормативов ООО "БВК" (ПГ КОНАР, г. Челябинск) и ГОСТ 977-88.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Базовые расценки шихты и легирующих элементов (руб / кг годного металла)
STEEL_PRICES: Dict[str, float] = {
    "20Л": 115.0,
    "25Л": 120.0,
    "35Л": 125.0,
    "20ГЛ": 140.0,
    "20ХМЛ": 185.0,
    "15Х1М1ФЛ": 240.0,
    "08Х18Н10ТЛ": 520.0,
    "12Х18Н12М3ТЛ": 640.0,
    "ВЧ 50": 95.0,
    "СЧ 20": 75.0,
    "DEFAULT": 130.0,
}

# Коэффициент выхода годного (КИМ) по типам деталей
YIELD_RATIOS: Dict[str, float] = {
    "flange": 0.65,      # Диски, фланцы (простая заливка, малые прибыли)
    "valve": 0.50,       # Клапаны, корпуса арматуры (сложная геометрия, крупные прибыли)
    "housing": 0.52,     # Корпусные детали редукторов, картеры
    "shaft": 0.62,       # Валы, оси
    "bracket": 0.55,     # Кронштейны, рычаги
    "DEFAULT": 0.55,
}

# Стоимость термической обработки (руб / кг отливки)
HEAT_TREATMENT_RATES: Dict[str, float] = {
    "отжиг": 14.0,
    "нормализация": 22.0,
    "нормализация_отпуск": 28.0,
    "закалка_отпуск": 38.0,
    "аустенизация": 55.0,
    "DEFAULT": 25.0,
}

# Стоимость контроля качества по группам отливок ГОСТ 977-88 (руб)
QUALITY_CONTROL_RATES: Dict[str, Dict[str, any]] = {
    "1": {
        "name": "Группа 1 (Общее назначение)",
        "base_cost": 25000.0,
        "operations": ["ВИК 100%", "Механические испытания образцов на растяжение и ударный изгиб"],
        "scrap_risk": 0.03,  # 3% закладываемый риск брака
    },
    "2": {
        "name": "Группа 2 (Ответственные детали)",
        "base_cost": 55000.0,
        "operations": ["ВИК 100%", "Механические испытания", "Магнитопорошковый контроль (МПД) сопряжений 20%"],
        "scrap_risk": 0.05,
    },
    "3": {
        "name": "Группа 3 (Особо ответственные детали)",
        "base_cost": 110000.0,
        "operations": ["ВИК 100%", "Механические испытания", "УЗК 100% зон сопряжений стенок", "МПД 100% наружной поверхности"],
        "scrap_risk": 0.08,
    },
    "3а": {
        "name": "Группа 3а (Детали под давлением, АЭС, РМРС)",
        "base_cost": 220000.0,
        "operations": [
            "ВИК 100% по РД 03-606-03",
            "УЗК 100% по ГОСТ 24507-80 (2 группа качества)",
            "Радиографический контроль (РК) по ГОСТ 23055-78",
            "Капиллярный контроль (ЦД) кромок под сварку",
            "Гидравлические испытания на герметичность Pпр"
        ],
        "scrap_risk": 0.12,  # 12% риск брака из-за строгой дефектоскопии
    }
}


@dataclass
class CostBreakdown:
    mass_part_kg: float
    mass_melt_kg: float
    steel_grade: str
    casting_group: str
    metal_cost: float
    melting_energy_cost: float
    tooling_pattern_cost: float
    molding_sand_cost: float
    heat_treatment_cost: float
    quality_control_cost: float
    machining_rough_cost: float
    scrap_risk_cost: float
    overhead_cost: float
    total_cost_expected: float
    total_cost_min: float
    total_cost_max: float
    operations_list: List[str]


class FoundryCostEngine:
    def __init__(self, energy_per_ton_kwh: float = 650.0, kwh_rate: float = 6.2, overhead_rate: float = 0.18):
        self.energy_per_ton = energy_per_ton_kwh
        self.kwh_rate = kwh_rate
        self.overhead_rate = overhead_rate

    def calculate(self,
                  mass_kg: float,
                  steel_grade: str = "20ГЛ",
                  casting_group: str = "3",
                  heat_treatment: str = "нормализация_отпуск",
                  part_type: str = "DEFAULT",
                  has_existing_pattern: bool = False) -> CostBreakdown:
        """
        Калькуляция полной себестоимости производства отливки.
        """
        if mass_kg <= 0:
            mass_kg = 50.0

        # Нормализация наименования стали
        norm_steel = "DEFAULT"
        for grade in STEEL_PRICES:
            if grade in steel_grade.upper():
                norm_steel = grade
                break
        steel_price_kg = STEEL_PRICES[norm_steel]

        # Расчет массы жидкого металла (с учетом прибылей и литников)
        yield_ratio = YIELD_RATIOS.get(part_type.lower(), YIELD_RATIOS["DEFAULT"])
        mass_melt_kg = mass_kg / yield_ratio

        # 1. Стоимость металла шихты и ферросплавов
        metal_cost = mass_melt_kg * steel_price_kg

        # 2. Плавка в дуговой / индукционной печи (электроэнергия + электроды + футеровка)
        melting_energy_cost = (mass_melt_kg / 1000.0) * (self.energy_per_ton * self.kwh_rate) * 1.35

        # 3. Модельная оснастка (Wood/MDF/Plastic CNC)
        if has_existing_pattern:
            # Адаптация существующей модельной оснастки аналога
            tooling_pattern_cost = 25000.0 + (mass_kg ** 0.5) * 1500.0
        else:
            # Изготовление новой деревянно-пластиковой оснастки с нуля
            tooling_pattern_cost = 75000.0 + (mass_kg ** 0.6) * 4500.0

        # 4. Формовка ХТС (холоднотвердеющие смоляные смеси) и стержни
        sand_weight_ton = (mass_melt_kg * 4.2) / 1000.0
        molding_sand_cost = sand_weight_ton * 14500.0  # песок + смола + катализатор

        # 5. Термообработка
        ht_key = "DEFAULT"
        for k in HEAT_TREATMENT_RATES:
            if k in heat_treatment.lower():
                ht_key = k
                break
        heat_treatment_cost = mass_kg * HEAT_TREATMENT_RATES[ht_key]

        # 6. Контроль качества по ГОСТ 977-88
        grp_key = str(casting_group).replace("группа", "").strip().lower()
        if grp_key not in QUALITY_CONTROL_RATES:
            grp_key = "3"
        qc_info = QUALITY_CONTROL_RATES[grp_key]
        quality_control_cost = qc_info["base_cost"] + (mass_kg ** 0.5) * 1200.0
        scrap_rate = qc_info["scrap_risk"]

        # 7. Черновая механическая обдирка отливки
        machining_rough_cost = mass_kg * 42.0

        # Прямые производственные затраты
        direct_costs = (metal_cost + melting_energy_cost + tooling_pattern_cost +
                        molding_sand_cost + heat_treatment_cost + quality_control_cost +
                        machining_rough_cost)

        # Риск брака при испытаниях
        scrap_risk_cost = direct_costs * scrap_rate

        # Общезаводские накладные расходы (OPEX, лаборатория, амортизация)
        overhead_cost = (direct_costs + scrap_risk_cost) * self.overhead_rate

        total_expected = direct_costs + scrap_risk_cost + overhead_cost
        total_min = total_expected * 0.90
        total_max = total_expected * 1.15

        # Список обязательных технологических операций
        ops = [
            f"Входной контроль шихтовых материалов и ферросплавов ({norm_steel})",
            f"Формовка ХТС (масса песчано-смоляной смеси ~{sand_weight_ton:.1f} т)",
            f"Плавка и раскисление жидкой стали в ДСП/ИП ({mass_melt_kg:.1f} кг расплава)",
            f"Заливка формы при температуре ~1560-1590°C",
            f"Охлаждение, выбивка отливки, удаление литниково-питающей системы (ЛПС)",
            f"Термическая обработка: {heat_treatment} ({mass_kg:.1f} кг)",
            f"Черновая обдирка базовых поверхностей и снятие припусков",
        ] + [f"Контроль качества: {op}" for op in qc_info["operations"]]

        return CostBreakdown(
            mass_part_kg=round(mass_kg, 1),
            mass_melt_kg=round(mass_melt_kg, 1),
            steel_grade=norm_steel,
            casting_group=qc_info["name"],
            metal_cost=round(metal_cost),
            melting_energy_cost=round(melting_energy_cost),
            tooling_pattern_cost=round(tooling_pattern_cost),
            molding_sand_cost=round(molding_sand_cost),
            heat_treatment_cost=round(heat_treatment_cost),
            quality_control_cost=round(quality_control_cost),
            machining_rough_cost=round(machining_rough_cost),
            scrap_risk_cost=round(scrap_risk_cost),
            overhead_cost=round(overhead_cost),
            total_cost_expected=round(total_expected),
            total_cost_min=round(total_min),
            total_cost_max=round(total_max),
            operations_list=ops
        )
