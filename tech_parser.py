# -*- coding: utf-8 -*-
"""
Модуль извлечения инженерных реквизитов штампа ГОСТ 2.104 и технических требований.
Проект: Оценка стоимости уникального литого изделия (ЮУрГУ / ООО "БВК")
"""

import re
from typing import Dict, Any, Optional

REGEX_MASS = re.compile(r"(?:масса|вес|m|м)\s*[:=\-]?\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:кг|к|т)?\b|\b([0-9]+(?:[\.,][0-9]+)?)\s*(?:кг|т)\b", re.I)
REGEX_STEEL = re.compile(r"\b(20Л|25Л|30Л|35Л|45Л|55Л|20ГЛ|20ХМЛ|10Г2СЛ|15Х1М1ФЛ|25Х1МФЛ|12Х1МФЛ|08Х18Н10ТЛ|12Х18Н12М3ТЛ|20Х25Н19С2Л|ВЧ\s*[3-8]0|СЧ\s*[1-3][05]|сталь\s*\d+[А-Яа-я0-9]*Л?)\b", re.I)
REGEX_GROUP = re.compile(r"\b(?:групп[а-я]\s*(?:отливки)?\s*([1-3][а-я]?|I{1,3}[а-я]?)|([1-3][а-я])\s*групп[а-я])\b", re.I)
REGEX_NDT = re.compile(r"\b(УЗК|РК|МПД|ЦД|ВИК|рентген\w*|радиографи\w*|ультразвук\w*|капиллярн\w*)\b", re.I)
REGEX_HEAT = re.compile(r"\b(нормализац\w+|отжиг\w*|закалк\w+|отпуск\w*|аустенизац\w+|улучшени\w+)\b", re.I)
REGEX_PRESSURE = re.compile(r"\b(?:гидроиспытан\w+|Pпр\s*=\s*([0-9\.]+)\s*МПа|давлен\w+\s*([0-9\.]+)\s*МПа)\b", re.I)

def parse_technical_text(raw_text: str) -> Dict[str, Any]:
    """
    Парсит сырой текст штампа или ТТ и извлекает структурированные параметры для калькулятора.
    Возвращает None для параметров, которые не были явно найдены в тексте (для активации эвристик).
    """
    res = {
        "steel_grade": None,
        "mass_kg": None,
        "casting_group": None,
        "heat_treatment": None,
        "ndt_methods": [],
        "pressure_test_mpa": None,
        "raw_matches": {}
    }

    if not raw_text:
        return res

    # 1. Поиск стали
    steel_match = REGEX_STEEL.search(raw_text)
    if steel_match:
        val = steel_match.group(1).replace(" ", "").upper().replace("СТАЛЬ", "")
        res["steel_grade"] = val if val else "20ГЛ"
        res["raw_matches"]["steel"] = steel_match.group(0)

    # 2. Поиск массы
    mass_match = REGEX_MASS.search(raw_text)
    if mass_match:
        raw_val = mass_match.group(1) or mass_match.group(2)
        try:
            val = float(raw_val.replace(",", "."))
            # Проверка единиц измерения: тонны
            match_str = raw_text[mass_match.start():mass_match.end()+6].lower()
            if " т" in match_str or "т." in match_str:
                val *= 1000.0
            if 0.2 <= val <= 100000:
                res["mass_kg"] = val
                res["raw_matches"]["mass"] = f"{val} кг"
        except (ValueError, TypeError):
            pass

    # 3. Поиск группы отливки ГОСТ 977-88
    grp_match = REGEX_GROUP.search(raw_text)
    if grp_match:
        grp = grp_match.group(1) or grp_match.group(2)
        grp = grp.replace("I", "1").replace("II", "2").replace("III", "3").lower()
        res["casting_group"] = grp
        res["raw_matches"]["group"] = f"Группа {grp}"

    # 4. Методы неразрушающего контроля
    ndt_matches = list(set(REGEX_NDT.findall(raw_text)))
    if ndt_matches:
        res["ndt_methods"] = ndt_matches
        res["raw_matches"]["ndt"] = ndt_matches
        # Если есть УЗК или РК, то группа обычно не ниже 3
        if ("УЗК" in ndt_matches or "РК" in ndt_matches) and res["casting_group"] in ("1", "2"):
            res["casting_group"] = "3"

    # 5. Термообработка
    ht_matches = list(set(REGEX_HEAT.findall(raw_text)))
    if ht_matches:
        joined = " ".join(ht_matches).lower()
        if "закалк" in joined:
            res["heat_treatment"] = "закалка_отпуск"
        elif "аустениз" in joined:
            res["heat_treatment"] = "аустенизация"
        elif "отжиг" in joined:
            res["heat_treatment"] = "отжиг"
        else:
            res["heat_treatment"] = "нормализация_отпуск"
        res["raw_matches"]["heat_treatment"] = ht_matches

    # 6. Гидроиспытания
    p_match = REGEX_PRESSURE.search(raw_text)
    if p_match:
        p_val = p_match.group(1) or p_match.group(2)
        if p_val:
            try:
                res["pressure_test_mpa"] = float(p_val)
                res["raw_matches"]["pressure"] = f"Pпр = {p_val} МПа"
            except ValueError:
                pass
        # Детали под давлением всегда относятся к группе 3а
        res["casting_group"] = "3а"

    return res
