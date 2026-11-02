"""Шаг 3 (после finalize.py): уточнение классов по результатам визуальной проверки превью.

1. Переименование классов под фактическое содержимое:
   06_kronshteyny_opory -> 06_petli_kronshteyny (в MCB это в основном петли),
   07_rychagi           -> 07_rychagi_kryuki    (много крюков).
2. Исключение «стержней»: модели вне класса валов, у которых средний габарит < 6% наибольшего
   (проволока и прутки — на чертеже это две линии). В params.csv / inventory.csv помечаются
   split/status = excluded_rod, из split.csv удаляются.

Запуск: python relabel.py --out out
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RENAME = {"06_kronshteyny_opory": "06_petli_kronshteyny", "07_rychagi": "07_rychagi_kryuki"}
ROD_RATIO = 0.06

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
out = Path(ap.parse_args().out)

for sub in ("png_views", "pdf", "preview", "png_full"):
    for old, new in RENAME.items():
        for p in (out / sub / old, out / sub / f"{old}.png"):
            if p.exists():
                p.rename(p.with_name(p.name.replace(old, new)))


def ren(df):
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype("string")
            for old, new in RENAME.items():
                s = s.str.replace(old, new, regex=False)
            df[c] = s
    return df


params = ren(pd.read_csv(out / "params.csv"))
split = ren(pd.read_csv(out / "split.csv"))
inv = ren(pd.read_csv(out / "inventory.csv"))

d = np.sort(params[["L", "W", "H"]].values.astype(float), axis=1)
params["mid_r"] = d[:, 1] / d[:, 2]
rod = (params.mid_r < ROD_RATIO) & (params.our_class != "04_valy_osi")
params.loc[rod, "split"] = "excluded_rod"
inv.loc[inv.designation.isin(params.loc[rod, "designation"]), "status"] = "excluded_rod"
split = split[~split.file_view.isin(params.loc[rod, "file_view"])]

params.to_csv(out / "params.csv", index=False, encoding="utf-8-sig")
split.to_csv(out / "split.csv", index=False)
inv.to_csv(out / "inventory.csv", index=False, encoding="utf-8-sig")
ok = params[params.split != "excluded_rod"]
tab = pd.crosstab(ok.our_class, ok.split); tab["total"] = tab.sum(axis=1)
print(f"исключено стержней: {int(rod.sum())}; итого в датасете: {len(ok)}\n{tab}")
