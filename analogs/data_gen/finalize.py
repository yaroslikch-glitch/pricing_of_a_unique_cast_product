"""Дедупликация, разбиение и превью после eskd_gen.py.
Запуск: python finalize.py --out out
Создаёт: out/params.csv, out/split.csv, out/inventory.csv, out/preview/<класс>.png, out/summary.txt
"""
import argparse, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--thr", type=float, default=0.97); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); out = Path(a.out)

df = pd.read_csv(out / "inventory_raw.csv")
ok = df[df.status == "ok"].copy()
ok = ok[ok.file_view.map(lambda p: isinstance(p, str) and (out / p).exists())].reset_index(drop=True)

# --- дубликаты внутри класса: почти одинаковые изображения видов
ok["dup_of"] = ""
def vec(p):
    a = np.asarray(Image.open(out / p).convert("L").resize((48, 48), Image.BILINEAR), float)
    a = 255 - a; a -= a.mean(); return a.ravel() / (np.linalg.norm(a) + 1e-9)
for cls, idx in ok.groupby("our_class").groups.items():
    idx = list(idx)
    F = np.stack([vec(ok.loc[i, "file_view"]) for i in idx]); S = F @ F.T
    removed = set()
    for i in range(len(idx)):
        if idx[i] in removed:
            continue
        for j in np.where(S[i, i + 1:] > a.thr)[0] + i + 1:
            if idx[j] not in removed:
                removed.add(idx[j]); ok.loc[idx[j], "dup_of"] = ok.loc[idx[i], "designation"]
df_dup = ok[ok.dup_of != ""]
ok = ok[ok.dup_of == ""].copy()

# --- разбиение 70/15/15 по классам
rng = random.Random(a.seed)
ok["split"] = ""
for cls, g in ok.groupby("our_class"):
    ids = list(g.index); rng.shuffle(ids)
    n = len(ids); ntr = round(n * 0.70); nva = round(n * 0.15)
    ok.loc[ids[:ntr], "split"] = "train"; ok.loc[ids[ntr:ntr + nva], "split"] = "val"
    ok.loc[ids[ntr + nva:], "split"] = "test"

cols = ["designation", "our_class", "split", "file_view", "file_pdf", "name", "material", "mass_kg",
        "scale", "format", "L", "W", "H", "revolution", "has_section", "mcb_source",
        "mass_casting_fact_SYNTH", "labor_hours_fact_SYNTH"]
ok[cols].to_csv(out / "params.csv", index=False, encoding="utf-8-sig")
ok[["file_view", "our_class", "split"]].to_csv(out / "split.csv", index=False)

inv = df.drop(columns=["fp"]).copy()
inv.loc[inv.designation.isin(df_dup.designation), "status"] = "duplicate"
inv.to_csv(out / "inventory.csv", index=False, encoding="utf-8-sig")

# --- превью 6x6 на класс
(out / "preview").mkdir(exist_ok=True)
for cls, g in ok.groupby("our_class"):
    files = list(g.file_view)[:36]
    sheet = Image.new("L", (6 * 160, 6 * 160), 255)
    for k, f in enumerate(files):
        sheet.paste(Image.open(out / f).convert("L").resize((156, 156)), ((k % 6) * 160 + 2, (k // 6) * 160 + 2))
    sheet.save(out / "preview" / f"{cls}.png")

# --- сводка
tab = pd.crosstab(ok.our_class, ok.split)
tab["total"] = tab.sum(axis=1)
reasons = df[df.status != "ok"].reason.str.split(":").str[0].value_counts()
txt = (f"Моделей на входе: {len(df)}\nУспешно: {(df.status == 'ok').sum()}\n"
       f"Удалено дубликатов: {len(df_dup)}\nИтого в датасете: {len(ok)}\n\n"
       f"Ошибки:\n{reasons.to_string()}\n\nПо классам:\n{tab.to_string()}\n")
(out / "summary.txt").write_text(txt, encoding="utf-8"); print(txt)
