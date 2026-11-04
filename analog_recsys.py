# -*- coding: utf-8 -*-
"""
Модуль рекомендательной системы подбора литых деталей-аналогов (RecSys Pipeline).
Спроектирован по архитектурному шаблону `recsys-pipeline-architect` (Source -> Hydrator -> Filter -> Scorer -> Selector -> SideEffect).
Интегрирован с энкодером визуальных признаков DINOv2 (facebookresearch/dinov2, 384d).
Проект: Оценка себестоимости уникального литья (ЮУрГУ & ООО «БВК» / ПГ КОНАР).
"""

import os
import math
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Каталог исторических деталей завода (прецедентная база ООО "БВК")
DEFAULT_CATALOG_PATH = Path("casting_catalog.json")
DEFAULT_EMBEDDINGS_PATH = Path("catalog_embeddings.npz")

# Базовые классы классификатора ЕСКД
ESKD_CLASSES = {
    "01": "Диски и фланцы",
    "02": "Колёса и шкивы",
    "03": "Втулки и стаканы",
    "04": "Валы и оси",
    "05": "Корпусные детали и арматура",
    "06": "Кронштейны, проушины и петли",
    "07": "Рычаги, тяги и крюки",
    "08": "Фасонные и опорные плиты"
}

_DINOV2_MODEL = None
_DINOV2_TRANSFORM = None

def get_dinov2_extractor():
    """Синглтон ленивой загрузки легковесной модели DINOv2 (ViT-S/14)."""
    global _DINOV2_MODEL, _DINOV2_TRANSFORM
    if _DINOV2_MODEL is None:
        try:
            import torch
            import torchvision.transforms as T
            _DINOV2_MODEL = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            _DINOV2_MODEL.eval()
            _DINOV2_TRANSFORM = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"[RecSys DINOv2] Предупреждение: DINOv2 не загружен ({e}), используется геометрический скоринг.")
            _DINOV2_MODEL = False
    return _DINOV2_MODEL, _DINOV2_TRANSFORM

def extract_visual_embedding(views_crop) -> Optional[np.ndarray]:
    """Извлекает нормализованный 384-мерный вектор визуальных признаков проекций."""
    model, transform = get_dinov2_extractor()
    if not model or views_crop is None:
        return None
    try:
        import torch
        from PIL import Image
        if isinstance(views_crop, np.ndarray):
            if len(views_crop.shape) == 2:
                pil_img = Image.fromarray(views_crop).convert("RGB")
            elif views_crop.shape[2] == 3:
                pil_img = Image.fromarray(views_crop)
            else:
                pil_img = Image.fromarray(views_crop[:, :, :3])
        elif isinstance(views_crop, Image.Image):
            pil_img = views_crop.convert("RGB")
        else:
            return None

        t = transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            feat = model(t).squeeze(0).numpy()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat
    except Exception as e:
        print(f"[RecSys] Ошибка извлечения DINOv2 эмбеддинга: {e}")
        return None

class RecsysAnalogMatcher:
    """
    Шестиэтапный конвейер поиска и ранжирования деталей-аналогов:
    1. Source: Загрузка базы выполненных отливок БВК и прекомпилированных эмбеддингов
    2. Hydrator: Обогащение металлургическими коэффициентами и геометрией
    3. Filter: Исключение технологически несопоставимых позиций
    4. Scorer: Мультимодальный скоринг (DINOv2 косинусное сходство + масса + материал + группа)
    5. Selector: Выборка Top-K аналогов
    6. SideEffect: Расчет ценовой дельты, калибровка себестоимости и экономия на оснастке
    """
    def __init__(self, catalog_path: Path = DEFAULT_CATALOG_PATH,
                 embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH):
        self.catalog_path = catalog_path
        self.embeddings_path = embeddings_path
        self.catalog = self._load_catalog()
        self.embeddings = self._load_embeddings()

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if self.catalog_path.exists():
            try:
                return json.loads(self.catalog_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[RecSys] Ошибка чтения {self.catalog_path}: {e}")
        return []

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        if self.embeddings_path.exists():
            try:
                data = np.load(self.embeddings_path)
                return {k: data[k] for k in data.files}
            except Exception as e:
                print(f"[RecSys] Ошибка чтения {self.embeddings_path}: {e}")
        return {}

    # 1. Source Stage
    def get_source_items(self) -> List[Dict[str, Any]]:
        return list(self.catalog)

    # 2. Hydrator Stage
    def hydrate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_copy = dict(item)
        mass = float(item_copy.get("mass_kg", 50.0))
        # Расчетная масса расплава с учетом технологического КИМ 0.55
        item_copy["melt_mass_kg"] = round(mass / 0.55, 1)
        dims = item_copy.get("dimensions_mm", [300, 300, 100])
        item_copy["bounding_volume_dm3"] = round((dims[0] * dims[1] * dims[2]) / 1_000_000, 2)
        return item_copy

    # 3. Filter Stage
    def filter_items(self, items: List[Dict[str, Any]], query_mass: float) -> List[Dict[str, Any]]:
        """Отсекает изделия, отличающиеся по массе более чем в 10 раз."""
        valid = []
        for it in items:
            m = it.get("mass_kg", 50.0)
            if query_mass > 0:
                ratio = max(m / query_mass, query_mass / m)
                if ratio > 10.0:
                    continue
            valid.append(it)
        return valid if valid else items

    # 4. Scorer Stage
    def score_items(self, items: List[Dict[str, Any]],
                    query_mass: float,
                    query_steel: str,
                    query_group: str,
                    query_aspect_ratio: float = 1.0,
                    query_embedding: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Мультимодальный расчет сходства:
        - Visual score (DINOv2 косинусное сходство 384d векторов или геометрический форм-фактор)
        - Mass score (близость массы по логарифмической шкале)
        - Material score (совпадение марки стали / легирования)
        - Group score (совпадение группы контроля ГОСТ 977-88)
        """
        scored = []
        has_dinov2 = query_embedding is not None and len(self.embeddings) > 0

        for it in items:
            it_id = it.get("id", "")
            m = it.get("mass_kg", 50.0)
            dims = it.get("dimensions_mm", [300, 300, 100])
            aspect = dims[0] / max(1.0, dims[1])

            # 1. Оценка близости масс: exp(-|ln(m1/m2)|)
            if query_mass > 0 and m > 0:
                s_mass = math.exp(-abs(math.log(query_mass / m)))
            else:
                s_mass = 0.5

            # 2. Оценка марки стали
            it_steel = it.get("steel_grade", "20Л").upper()
            q_steel = query_steel.upper()
            if it_steel == q_steel:
                s_mat = 1.0
            elif ("Л" in it_steel and "Л" in q_steel) or ("ГЛ" in it_steel and "ГЛ" in q_steel):
                s_mat = 0.8
            else:
                s_mat = 0.4

            # 3. Визуальный скор (DINOv2)
            if has_dinov2 and it_id in self.embeddings:
                c_emb = self.embeddings[it_id]
                cos_sim = float(np.dot(query_embedding, c_emb))
                # Переводим косинус [-1.0; 1.0] в шкалу [0.0; 1.0] с акцентом на высокую корреляцию
                s_vis = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))
            else:
                s_vis = math.exp(-abs(aspect - query_aspect_ratio))

            # 4. Оценка группы отливки (ГОСТ 977-88)
            it_grp = str(it.get("casting_group", "3"))
            q_grp = str(query_group)
            s_grp = 1.0 if it_grp == q_grp else 0.7

            # Интегральный скор с весами
            if has_dinov2:
                # DINOv2 доступен: 35% визуал, 35% масса, 20% марка стали, 10% группа
                total_score = round(0.35 * s_vis + 0.35 * s_mass + 0.20 * s_mat + 0.10 * s_grp, 3)
            else:
                # Геометрический скоринг: 40% масса, 30% материал, 20% соотношение сторон, 10% группа
                total_score = round(0.40 * s_mass + 0.30 * s_mat + 0.20 * s_vis + 0.10 * s_grp, 3)

            scored_entry = dict(it)
            scored_entry["similarity_score"] = total_score
            scored_entry["sim_pct"] = round(total_score * 100, 1)
            scored_entry["sim_visual_pct"] = round(s_vis * 100, 1)
            scored_entry["sim_mass_pct"] = round(s_mass * 100, 1)
            scored_entry["sim_material_pct"] = round(s_mat * 100, 1)
            scored_entry["sim_group_pct"] = round(s_grp * 100, 1)
            
            # Экономия на модельной оснастке
            pattern_savings = it.get("pattern_cost_rub", 0) if it.get("has_pattern") else 0
            scored_entry["pattern_savings_rub"] = pattern_savings

            scored.append(scored_entry)

        return scored

    # 5. Selector Stage
    def select_top_k(self, scored_items: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
        scored_items.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_items[:k]

    # 6. SideEffect Stage (Case-Based Reasoning)
    def apply_side_effects(self, top_k: List[Dict[str, Any]], calculated_cost: float) -> Dict[str, Any]:
        """Сопоставляет расчетную стоимость с фактическими прецедентами БВК."""
        if not top_k:
            return {
                "top_analogs": [],
                "weighted_analog_cost": calculated_cost,
                "cost_delta_rub": 0,
                "cost_delta_pct": 0.0,
                "pattern_savings_max_rub": 0,
                "best_analog": None,
                "recommendation": "Аналогов в базе завода не обнаружено."
            }

        weights = [it["similarity_score"] for it in top_k]
        w_sum = sum(weights) if sum(weights) > 0 else 1.0
        
        hist_costs = [it["actual_cost_rub"] for it in top_k]
        weighted_cost = sum(c * w for c, w in zip(hist_costs, weights)) / w_sum

        cost_delta = calculated_cost - weighted_cost
        cost_delta_pct = (cost_delta / weighted_cost) * 100 if weighted_cost > 0 else 0.0

        best = top_k[0]
        pattern_savings = best.get("pattern_savings_rub", 0)

        # Формирование инженерной рекомендации CBR
        if best["sim_pct"] >= 75.0 and best.get("has_pattern"):
            rec_text = (
                f"Высокая степень подобия с {best['id']} ({best['sim_pct']}%). "
                f"На складе БВК имеется модельный комплект (экономия до {pattern_savings:,.0f} руб.). "
                f"Рекомендуется адаптация существующей оснастки вместо изготовления с нуля."
            )
        elif best["sim_pct"] >= 60.0:
            rec_text = (
                f"Умеренное сходство с изделием {best['id']} ({best['sim_pct']}%). "
                f"Рекомендуется использовать отработанную технологическую карту заливки и термообработки."
            )
        else:
            rec_text = (
                "Уникальная конфигурация. Требуется полный цикл проектирования литниково-питающей системы (ЛПС) "
                "и изготовление индивидуального модельного комплекта."
            )

        return {
            "top_analogs": top_k,
            "weighted_analog_cost": round(weighted_cost),
            "cost_delta_rub": round(cost_delta),
            "cost_delta_pct": round(cost_delta_pct, 1),
            "pattern_savings_max_rub": pattern_savings,
            "best_analog": best,
            "recommendation": rec_text
        }

    # Полный пайплайн поиска аналогов
    def find_analogs(self, mass_kg: float, steel_grade: str, casting_group: str,
                     aspect_ratio: float = 1.0, views_crop=None, k: int = 3) -> Dict[str, Any]:
        """
        Запускает сквозной конвейер RecSys:
        Source -> Hydrate -> Filter -> Score -> Select -> SideEffect.
        """
        # 1. Source
        items = self.get_source_items()
        
        # 2. Hydrate
        hydrated = [self.hydrate_item(it) for it in items]
        
        # 3. Filter
        filtered = self.filter_items(hydrated, query_mass=mass_kg)
        
        # Извлекаем DINOv2 эмбеддинг, если переданы проекции чертежа
        query_embedding = None
        if views_crop is not None:
            query_embedding = extract_visual_embedding(views_crop)

        # 4. Score
        scored = self.score_items(
            filtered,
            query_mass=mass_kg,
            query_steel=steel_grade,
            query_group=casting_group,
            query_aspect_ratio=aspect_ratio,
            query_embedding=query_embedding
        )
        
        # 5. Select
        selected = self.select_top_k(scored, k=k)
        return selected
