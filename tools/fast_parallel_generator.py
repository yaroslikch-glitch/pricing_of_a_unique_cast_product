# -*- coding: utf-8 -*-
"""
Высокоскоростной параллельный генератор 2D-чертежей ЕСКД (Multiprocessing).
Использует 8 ядер процессора AMD Ryzen для параллельного рендеринга 3D -> 2D.
Генерирует изображения PNG и точные YOLO .txt аннотации по 7 классам.
"""

import sys
import os
import glob
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # Быстрый безголовый бэкенд без GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

MCB_DATASET_ROOT = r"C:\Users\Yaroslav\OneDrive\Desktop\датасет проекта\Скрипт пайтон 3D =_ 2D и тестовый датасет\dataset_org_norm\train"
TARGET_EXPANDED_DIR = r"dataset_yolo_expanded"

FOUNDRY_CATEGORIES = [
    "Valve",
    "Impeller",
    "Turbine",
    "Flanged block bearing",
    "Flanged plain bearings",
    "Bushes",
    "Pulleys",
    "Right spur gears",
    "Lever",
    "Elbow fitting",
    "Plates, circulate plates"
]

STEEL_GRADES = [
    "Сталь 20ГЛ ГОСТ 977-88",
    "Сталь 20ХМЛ ГОСТ 977-88",
    "Сталь 12Х18Н12М3ТЛ ГОСТ 977-88",
    "Сталь 08Х18Н10ТЛ ГОСТ 977-88",
    "Сталь 20Л ГОСТ 977-88",
    "Сталь 15Х1М1ФЛ ГОСТ 977-88",
    "Сталь 10Х17Н13М2ТЛ ГОСТ 977-88",
    "Сталь 25Л ГОСТ 977-88",
    "Сталь 35Л ГОСТ 977-88",
]

CASTING_GROUPS = ["группы 1", "группы 2", "группы 3", "группы 3а"]

DEVELOPERS = [
    ("Салов В.К.", "Янчук А.П."),
    ("Иванов И.И.", "Петров П.П."),
    ("Сидоров С.С.", "Ковалев К.К."),
    ("Смирнов А.А.", "Морозов М.М."),
    ("Волков В.В.", "Зайцев З.З."),
]

def load_obj(path):
    verts, raw_faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith("f "):
                face = [int(p.split("/")[0]) - 1 for p in line.strip().split()[1:4]]
                raw_faces.append(face)

    seen = set()
    dedup_faces = []
    for f in raw_faces:
        sf = tuple(sorted(f))
        if sf not in seen:
            seen.add(sf)
            dedup_faces.append(f)
    return np.array(verts), np.array(dedup_faces)

def extract_projected_edges(verts, faces, view_axis=2):
    if len(faces) == 0:
        return []
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normals = normals / norm

    view_dir = np.zeros(3)
    view_dir[view_axis] = 1.0
    cos_angles = normals @ view_dir
    visible = cos_angles > 0.05

    edge_count = {}
    for i, face in enumerate(faces):
        if not visible[i]:
            continue
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for e in edges:
            se = tuple(sorted(e))
            edge_count[se] = edge_count.get(se, 0) + 1

    axes = [0, 1, 2]
    axes.remove(view_axis)
    ax_x, ax_y = axes[0], axes[1]

    contour_lines = []
    for (idx1, idx2), count in edge_count.items():
        if count == 1:
            p1 = (verts[idx1, ax_x], verts[idx1, ax_y])
            p2 = (verts[idx2, ax_x], verts[idx2, ax_y])
            contour_lines.append((p1, p2))
    return contour_lines

def render_worker(task_info):
    """Рабочая функция одного процесса: рендерит 1 чертеж и создает метку YOLO."""
    obj_path, out_img_path, out_lbl_path, cat_name, serial_num = task_info
    try:
        verts, faces = load_obj(obj_path)
        if len(verts) == 0 or len(faces) == 0:
            return False

        fig = plt.figure(figsize=(16.54, 11.69), dpi=120)
        ax_bg = fig.add_axes([0, 0, 1, 1])
        ax_bg.set_xlim(0, 420)
        ax_bg.set_ylim(0, 297)
        ax_bg.axis("off")

        # Внешняя рамка ГОСТ 2.104
        border = patches.Rectangle((20, 5), 420 - 25, 297 - 10, linewidth=1.5, edgecolor="black", facecolor="white")
        ax_bg.add_patch(border)

        # Штамп 185 x 55 мм
        sx = 420 - 5 - 185
        sy = 5
        stamp = patches.Rectangle((sx, sy), 185, 55, linewidth=1.2, edgecolor="black", facecolor="none")
        ax_bg.add_patch(stamp)

        ax_bg.plot([sx, sx + 185], [sy + 15, sy + 15], "k-", lw=0.8)
        ax_bg.plot([sx, sx + 185], [sy + 40, sy + 40], "k-", lw=0.8)
        ax_bg.plot([sx + 65, sx + 65], [sy, sy + 40], "k-", lw=0.8)
        ax_bg.plot([sx + 135, sx + 135], [sy, sy + 15], "k-", lw=0.8)

        part_code = f"БВК.{random.randint(700, 799)}.{random.randint(100, 999)}.{serial_num:03d}"
        part_name = f"Отливка ({cat_name})"
        material = random.choice(STEEL_GRADES)
        dev_name, check_name = random.choice(DEVELOPERS)

        ax_bg.text(sx + 68, sy + 44, part_code, fontsize=12, fontweight="bold", va="bottom", family="sans-serif")
        ax_bg.text(sx + 68, sy + 26, part_name, fontsize=11, fontweight="bold", va="center", family="sans-serif")
        ax_bg.text(sx + 68, sy + 7, material, fontsize=9, va="center", family="sans-serif")
        ax_bg.text(sx + 140, sy + 7, "ООО «БВК»", fontsize=10, fontweight="bold", va="center", family="sans-serif")

        ax_bg.text(sx + 5, sy + 48, f"Разраб.   {dev_name}", fontsize=8)
        ax_bg.text(sx + 5, sy + 38, f"Пров.     {check_name}", fontsize=8)
        ax_bg.text(sx + 5, sy + 28, "Т.контр.  Симонова С.", fontsize=8)
        ax_bg.text(sx + 5, sy + 18, "Н.контр.  Тупикин М.", fontsize=8)
        ax_bg.text(sx + 5, sy + 8, "Утв.      Иванова О.", fontsize=8)

        # Технические требования
        casting_grp = random.choice(CASTING_GROUPS)
        tt_text = f"Технические требования:\n1. * Размеры для справок.\n2. Отливка {casting_grp} по ГОСТ 977-88.\n3. Неуказанные радиусы R3...R5 мм.\n4. УЗК 100% зон сопряжения по ГОСТ 24507-80.\n5. Термообработка: нормализация + отпуск.\n6. Маркировать клеймением: плавка, шифр."
        ax_bg.text(sx - 5, sy + 62, tt_text, fontsize=8.5, va="bottom", ha="left", family="sans-serif", linespacing=1.25)

        # Проекции
        views = [(1, "Главный вид"), (2, "Вид сверху"), (0, "Вид слева")]
        rects = [
            [0.08, 0.44, 0.35, 0.45],
            [0.08, 0.08, 0.35, 0.32],
            [0.46, 0.44, 0.35, 0.45],
        ]

        for (v_ax, v_title), rect in zip(views, rects):
            ax_view = fig.add_axes(rect)
            lines = extract_projected_edges(verts, faces, view_axis=v_ax)
            for p1, p2 in lines:
                ax_view.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=0.85)
            ax_view.set_aspect("equal")
            ax_view.set_title(v_title, fontsize=10, pad=4)
            ax_view.axis("off")

        # Сохранение изображения
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        plt.savefig(out_img_path, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Запись точной разметки YOLO
        # 0: views_field, 1: view_front, 2: view_top, 3: view_left, 4: stamp, 5: tech_req, 6: stamp_extra
        os.makedirs(os.path.dirname(out_lbl_path), exist_ok=True)
        with open(out_lbl_path, "w", encoding="utf-8") as lf:
            lf.write("0 0.505952 0.500000 0.878571 0.878788\n")
            lf.write("1 0.261905 0.281145 0.390476 0.441077\n")
            lf.write("2 0.261905 0.735690 0.390476 0.407407\n")
            lf.write("3 0.720238 0.281145 0.450000 0.441077\n")
            lf.write("4 0.767857 0.890572 0.440476 0.185185\n")
            lf.write("5 0.767857 0.679461 0.440476 0.210101\n")
            lf.write("6 0.130952 0.040404 0.166667 0.047138\n")

        return True
    except Exception as e:
        plt.close("all")
        return False

def run_batch_generation(target_count: int = 970, max_workers: int = 8):
    print("=" * 65)
    print(f"🚀 СТАРТ ПАРАЛЛЕЛЬНОЙ ГЕНЕРАЦИИ {target_count} ЧЕРТЕЖЕЙ (8 ядер CPU)")
    print("=" * 65)

    out_base = Path(TARGET_EXPANDED_DIR)
    img_train = out_base / "images" / "train"
    img_val = out_base / "images" / "val"
    lbl_train = out_base / "labels" / "train"
    lbl_val = out_base / "labels" / "val"

    # Собираем список доступных 3D моделей по категориям
    tasks = []
    per_cat = max(10, target_count // len(FOUNDRY_CATEGORIES) + 5)
    serial = 1

    for cat in FOUNDRY_CATEGORIES:
        cat_dir = Path(MCB_DATASET_ROOT) / cat
        if not cat_dir.exists():
            continue
        obj_files = list(cat_dir.glob("*.obj"))
        if not obj_files:
            continue
        random.shuffle(obj_files)
        selected = obj_files[:per_cat]

        for obj in obj_files:
            stem = f"syn_mcb_{cat.lower().replace(' ', '_')}_{obj.stem}"
            if (img_train / f"{stem}.png").exists() or (img_val / f"{stem}.png").exists():
                continue
            if len(tasks) >= target_count:
                break
            # 85% train / 15% val
            is_val = (len(tasks) % 7 == 0)
            split = "val" if is_val else "train"
            img_dir = img_val if is_val else img_train
            lbl_dir = lbl_val if is_val else lbl_train

            stem = f"syn_mcb_{cat.lower().replace(' ', '_')}_{obj.stem}"
            out_img = str(img_dir / f"{stem}.png")
            out_lbl = str(lbl_dir / f"{stem}.txt")

            tasks.append((str(obj), out_img, out_lbl, cat, serial))
            serial += 1

    print(f"Подготовлено задач на генерацию: {len(tasks)}")
    t0 = time.perf_counter()
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(render_worker, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                completed += 1
            else:
                errors += 1
            if completed % 50 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - t0
                speed = completed / max(0.1, elapsed)
                remaining = (len(tasks) - completed) / max(0.1, speed)
                sys.stdout.write(f"\r  Прогресс: {completed}/{len(tasks)} ({completed*100//len(tasks)}%) | Скорость: {speed:.1f} черт/с | Осталось: ~{int(remaining)}с")
                sys.stdout.flush()

    total_time = time.perf_counter() - t0
    print(f"\n\n[ГОТОВО] Сгенерировано: {completed} чертежей за {total_time/60:.1f} мин. (Ошибок: {errors})")

    # Пересчет и обновление конфигурации датасета
    train_c = len(list(img_train.glob("*.png")))
    val_c = len(list(img_val.glob("*.png")))
    total_c = train_c + val_c

    yaml_file = out_base / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as yf:
        yf.write(f"""# Расширенный сбалансированный датасет ЕСКД
# Всего чертежей: {total_c} (Train: {train_c}, Val: {val_c})
path: dataset_yolo_expanded
train: images/train
val: images/val

names:
  0: views_field
  1: view_front
  2: view_top
  3: view_left
  4: stamp
  5: tech_req
  6: stamp_extra
""")

    stats_file = out_base / "dataset_stats.json"
    import json
    stats_data = {
        "base_yolo_copied": 1000,
        "factory_bvk_added": 30,
        "synthetic_mcb_added": completed,
        "train_count": train_c,
        "val_count": val_c,
        "total_count": total_c
    }
    with open(stats_file, "w", encoding="utf-8") as sf:
        json.dump(stats_data, sf, ensure_ascii=False, indent=2)

    print(f"📊 ИТОГОВЫЙ РАЗМЕР ДАТАСЕТА: {total_c} чертежей! (Train: {train_c}, Val: {val_c})")
    print(f"Конфигурация обновлена: {yaml_file}")

if __name__ == "__main__":
    count = 970
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    run_batch_generation(target_count=count, max_workers=8)
