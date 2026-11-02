"""Шаг 1: отбор моделей из Mechanical Components Benchmark (MCB) по 8 классам литых деталей.

MCB (Kim et al., ECCV 2020): набор A — 68 классов, набор B — 25 классов, формат OBJ.
Ссылка: https://github.com/stnoah1/mcb . Модели используются только внутри проекта со ссылкой на авторов.

Сопоставление папок MCB нашим классам (классификатор ЕСКД, классы 71–74) и лимит N моделей
с каждой исходной папки. Первоначально отбор делался PowerShell-скриптом на Windows
(collect_msb.ps1); это эквивалентная кроссплатформенная версия.

Запуск: python select_mcb.py --a <MCB_A/dataset_org_norm> --b <MCB_B> --out mcb_raw [--n 120]
Результат: mcb_raw/<класс>/<A|B>_<папка>__<имя>.obj
"""
import argparse, random, shutil
from pathlib import Path

MAP = {
    "01_diski_flantsy":     [("B", "disc"), ("B", "flange")],
    "02_kolesa_shkivy":     [("A", "Pulleys"), ("A", "Wheel"), ("A", "Right spur gears"), ("A", "Impeller"),
                             ("B", "gear"), ("B", "pulley")],
    "03_vtulki_stakany":    [("A", "Bushes"), ("A", "Collars"), ("B", "bushing"), ("B", "ring")],
    "04_valy_osi":          [("B", "pin"), ("B", "rotor")],
    "05_korpusnye":         [("A", "Flanged block bearing"), ("A", "Valve")],
    "06_kronshteyny_opory": [("A", "Hinge"), ("B", "clamp"), ("B", "hinge")],
    "07_rychagi":           [("A", "Lever"), ("A", "Hook"), ("B", "fork_joint"), ("B", "hook")],
    "08_ploskie_fasonnye":  [("A", "Plates, circulate plates"), ("B", "plate")],
}

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--out", default="mcb_raw"); ap.add_argument("--n", type=int, default=120)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
roots = {"A": Path(a.a), "B": Path(a.b)}
rng = random.Random(a.seed)
for cls, sources in MAP.items():
    dst = Path(a.out) / cls; dst.mkdir(parents=True, exist_ok=True); total = 0
    for s, folder in sources:
        files = sorted(f for sub in ("train", "test") for f in (roots[s] / sub / folder).glob("*.obj"))
        rng.shuffle(files)
        tag = folder.replace(" ", "_").replace(",", "")
        for f in files[:a.n]:
            shutil.copy2(f, dst / f"{s}_{tag}__{f.name}")
        total += min(len(files), a.n)
        print(f"{cls:22s} {s}/{folder:28s} {min(len(files), a.n):4d} из {len(files)}")
    print(f"{cls}: {total}\n")
