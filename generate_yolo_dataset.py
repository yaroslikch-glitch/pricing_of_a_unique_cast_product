# -*- coding: utf-8 -*-
"""
Генератор синтетических чертежей ЕСКД с автоматической разметкой под YOLO.
Проект: Рекомендательный сервис для оценки стоимости изготовления уникальных литых изделий (ЮУрГУ / ООО "БВК").

Автоматически размечает классы:
  0: views_field  (всё поле проекционных видов целиком)
  1: view_front   (главный вид спереди)
  2: view_top     (вид сверху)
  3: view_left    (вид слева / сбоку)
  4: stamp        (основная надпись по ГОСТ 2.104 185х55 мм)
  5: tech_req     (технические требования)
  6: stamp_extra  (дополнительная графа 70х14 мм в левом верхнем углу)
"""

import os
import sys
import glob
import random
import argparse
import time
import concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Настройки чертежного шрифта и стилей линий
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "figure.autolayout": False
})

S_LW = 1.6  # Основная толстая линия (контуры, рамка)
T_LW = 0.6  # Тонкая линия (осевые, выносные, линии штампа)

# Справочные данные для штампа и ТТ (ООО "БВК", ПГ КОНАР, ЮУрГУ)
STEEL_GRADES = [
    "Сталь 20ГЛ ГОСТ 977-88",
    "Сталь 20ХМЛ ГОСТ 977-88",
    "Сталь 12Х18Н12М3ТЛ ГОСТ 977-88",
    "Сталь 08Х18Н10ТЛ ГОСТ 977-88",
    "Сталь 20Л ГОСТ 977-88",
    "Сталь 15Х1М1ФЛ ГОСТ 977-88",
    "Сталь 35Л ГОСТ 977-88",
    "Чугун ВЧ 50 ГОСТ 7293-85",
    "Чугун СЧ 20 ГОСТ 1412-85",
]

CASTING_GROUPS = ["группы 1", "группы 2", "группы 3", "группы 3а"]

NDT_REQUIREMENTS = [
    "УЗК 100% зон сопряжения стенок по ГОСТ 24507-80 (2 группа качества)",
    "Радиографический контроль (РК) фланцевых зон по ГОСТ 23055-78, класс 2",
    "Магнитопорошковый контроль (МПД) наружной поверхности по ГОСТ 21105-87, уровень Б",
    "Капиллярный контроль (ЦД) кромок под сварку по ГОСТ 18442-80, класс II",
    "Визуально-измерительный контроль (ВИК) по РД 03-606-03 в объеме 100%",
    "УЗК околошовных зон по ПНАЭ Г-7-014-89 для оборудования АЭС",
]

HEAT_TREATMENTS = [
    "Нормализация с последующим отпуском.",
    "Аустенизация с охлаждением в воде.",
    "Закалка и высокий отпуск (улучшение).",
    "Отжиг для снятия остаточных напряжений.",
]

PRESSURE_TESTS = [
    "Гидравлические испытания на прочность и плотность давлением Pпр = 2.5 МПа.",
    "Гидравлические испытания давлением Pпр = 4.0 МПа по ГОСТ 356-80.",
    "Гидравлические испытания давлением Pпр = 6.4 МПа. Выдержка не менее 10 минут.",
    "Пневматические испытания на герметичность воздухом P = 0.6 МПа под водой.",
]

CLASSES_NAMES = [
    "views_field",
    "view_front",
    "view_top",
    "view_left",
    "stamp",
    "tech_req",
    "stamp_extra"
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
    if len(verts) == 0 or len(raw_faces) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    return np.array(verts), np.array(raw_faces)

def extract_projected_edges(verts, faces, view_axis=2):
    if len(verts) == 0 or len(faces) == 0:
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

    edge_faces = {}
    for f_idx, f in enumerate(faces):
        for i in range(3):
            e = tuple(sorted([f[i], f[(i + 1) % 3]]))
            edge_faces.setdefault(e, []).append(f_idx)

    draw_edges = []
    cos_threshold = np.cos(np.radians(25))

    for e, f_list in edge_faces.items():
        if len(f_list) == 1:
            draw_edges.append(e)
        elif len(f_list) == 2:
            n1, n2 = normals[f_list[0]], normals[f_list[1]]
            if np.dot(n1, n2) < cos_threshold:
                draw_edges.append(e)
            else:
                dot1 = np.dot(n1, view_dir)
                dot2 = np.dot(n2, view_dir)
                if (dot1 > 0 and dot2 <= 0) or (dot1 <= 0 and dot2 > 0):
                    draw_edges.append(e)

    axes_2d = [i for i in range(3) if i != view_axis]
    lines_2d = []
    for e in draw_edges:
        p1 = verts[e[0]][axes_2d]
        p2 = verts[e[1]][axes_2d]
        lines_2d.append((p1, p2))
    return lines_2d

def to_yolo_bbox(xmin, ymin, xmax, ymax, sheet_w, sheet_h):
    """
    Конвертация координат миллиметров ЕСКД (origin снизу-слева)
    в нормализованный формат YOLO (origin сверху-слева, x_center, y_center, w, h).
    """
    # В экранных координатах изображения Y инвертирован (0 сверху)
    y_top = sheet_h - ymax
    y_bottom = sheet_h - ymin

    x_center = (xmin + xmax) / (2.0 * sheet_w)
    y_center = (y_top + y_bottom) / (2.0 * sheet_h)
    w = (xmax - xmin) / float(sheet_w)
    h = (ymax - ymin) / float(sheet_h)

    # Ограничение диапазоном 0..1
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.001, min(1.0, w))
    h = max(0.001, min(1.0, h))

    return x_center, y_center, w, h

def render_and_label_sheet(obj_path, out_img_path, out_txt_path, fmt="A3", preview_path=None, dpi=150):
    verts, faces = load_obj(obj_path)
    if len(verts) == 0:
        return False

    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    cat_name = os.path.basename(os.path.dirname(obj_path))

    # Размеры листа ЕСКД (мм)
    if fmt == "A4":
        W0, H0 = 210.0, 297.0
    else:
        W0, H0 = 420.0, 297.0

    fig = plt.figure(figsize=(W0 / 25.4, H0 / 25.4), dpi=dpi)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, W0)
    ax_bg.set_ylim(0, H0)
    ax_bg.set_aspect("equal")
    ax_bg.axis("off")

    # 1. Рамка листа (ГОСТ 2.104: 20 мм слева, 5 мм с остальных)
    frame_xmin, frame_ymin = 20.0, 5.0
    frame_xmax, frame_ymax = W0 - 5.0, H0 - 5.0
    border = patches.Rectangle(
        (frame_xmin, frame_ymin),
        frame_xmax - frame_xmin,
        frame_ymax - frame_ymin,
        linewidth=S_LW,
        edgecolor="black",
        facecolor="white"
    )
    ax_bg.add_patch(border)

    # 2. Дополнительная графа 70х14 в левом верхнем углу (stamp_extra)
    extra_xmin, extra_ymin = 20.0, H0 - 19.0
    extra_xmax, extra_ymax = 90.0, H0 - 5.0
    ax_bg.add_patch(patches.Rectangle((extra_xmin, extra_ymin), 70, 14, fill=False, lw=S_LW))
    part_code = f"БВК.{random.randint(710, 749)}.{random.randint(100, 999)}.{base_name[-3:]}"
    ax_bg.text(55, H0 - 12, part_code, fontsize=8, ha="center", va="center", rotation=180)

    # 3. Основная надпись / Штамп 185х55 мм (stamp)
    stamp_xmin = W0 - 5.0 - 185.0
    stamp_ymin = 5.0
    stamp_xmax = W0 - 5.0
    stamp_ymax = 60.0

    ax_bg.add_patch(patches.Rectangle((stamp_xmin, stamp_ymin), 185, 55, fill=False, lw=S_LW))
    # Внутренние линии штампа
    ax_bg.plot([stamp_xmin, stamp_xmax], [stamp_ymin + 15, stamp_ymin + 15], "k-", lw=T_LW)
    ax_bg.plot([stamp_xmin, stamp_xmax], [stamp_ymin + 40, stamp_ymin + 40], "k-", lw=S_LW)
    ax_bg.plot([stamp_xmin + 65, stamp_xmin + 65], [stamp_ymin, stamp_ymin + 40], "k-", lw=S_LW)
    ax_bg.plot([stamp_xmin + 135, stamp_xmin + 135], [stamp_ymin, stamp_ymin + 15], "k-", lw=T_LW)
    for off_x in [7, 17, 40, 55]:
        ax_bg.plot([stamp_xmin + off_x, stamp_xmin + off_x], [stamp_ymin, stamp_ymin + 55], "k-", lw=T_LW)

    # Текст штампа
    material = random.choice(STEEL_GRADES)
    part_name = f"Отливка ({cat_name})"
    ax_bg.text(stamp_xmin + 125, stamp_ymin + 47.5, part_code, fontsize=12, fontweight="bold", ha="center", va="center")
    ax_bg.text(stamp_xmin + 100, stamp_ymin + 27.5, part_name, fontsize=11, fontweight="bold", ha="center", va="center")
    ax_bg.text(stamp_xmin + 100, stamp_ymin + 7.5, material, fontsize=8.5, ha="center", va="center")
    ax_bg.text(stamp_xmin + 160, stamp_ymin + 7.5, "ООО «БВК»", fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax_bg.text(stamp_xmin + 5, stamp_ymin + 48, "Разраб.   Салов В.К.", fontsize=6.5)
    ax_bg.text(stamp_xmin + 5, stamp_ymin + 38, "Пров.     Янчук А.П.", fontsize=6.5)
    ax_bg.text(stamp_xmin + 5, stamp_ymin + 28, "Т.контр.  Симонова С.С.", fontsize=6.5)
    ax_bg.text(stamp_xmin + 5, stamp_ymin + 18, "Н.контр.  Тупикин М.Т.", fontsize=6.5)
    ax_bg.text(stamp_xmin + 5, stamp_ymin + 8,  "Утв.      Иванова О.Н.", fontsize=6.5)

    # 4. Технические требования (tech_req)
    selected_ndt = random.sample(NDT_REQUIREMENTS, k=random.randint(2, 4))
    ht = random.choice(HEAT_TREATMENTS)
    pt = random.choice(PRESSURE_TESTS)
    casting_grp = random.choice(CASTING_GROUPS)

    tt_lines = [
        "Технические требования:",
        "1. * Размеры для справок.",
        f"2. Отливка {casting_grp} по ГОСТ 977-88.",
        "3. Неуказанные литейные радиусы R3...R5 мм.",
        "4. Контроль неразрушающими методами:",
    ]
    for ndt in selected_ndt:
        tt_lines.append(f"   - {ndt};")
    tt_lines.extend([
        f"5. Термообработка: {ht}",
        f"6. {pt}",
        "7. Маркировать клеймением: номер плавки, шифр.",
    ])

    n_lines = len(tt_lines)
    tt_line_height = 5.2
    tt_height = (n_lines + 1) * tt_line_height
    tt_xmin = stamp_xmin
    tt_xmax = stamp_xmax
    tt_ymin = stamp_ymax + 4.0
    tt_ymax = tt_ymin + tt_height

    ax_bg.text(
        tt_xmin,
        tt_ymin,
        "\n".join(tt_lines),
        fontsize=7.5,
        va="bottom",
        ha="left",
        linespacing=1.25,
    )

    # 5. Проекционные виды (views)
    # Нормализуем координаты детали
    ext = verts.max(axis=0) - verts.min(axis=0)
    ext_max = max(ext.max(), 1e-6)

    # Доступная область под виды на листе А3: от x=25 до x=225 (левее штампа), y от 15 до 280
    # Координаты прямоугольников видов в мм на листе
    # [xmin, ymin, width, height]
    v_front_rect = [30.0, 150.0, 160.0, 125.0]
    v_top_rect   = [30.0, 20.0,  160.0, 115.0]
    v_left_rect  = [210.0, 150.0, 185.0, 125.0]

    views_spec = [
        ("view_front", 1, v_front_rect, "Главный вид"),
        ("view_top",   2, v_top_rect,   "Вид сверху"),
        ("view_left",  0, v_left_rect,  "Вид слева"),
    ]

    view_boxes = []

    for v_name, v_axis, (vx, vy, vw, vh), v_title in views_spec:
        # Добавляем оси для проекции
        # ax_view в нормализованных координатах фигуры [x/W0, y/H0, w/W0, h/H0]
        ax_view = fig.add_axes([vx / W0, vy / H0, vw / W0, vh / H0])
        lines = extract_projected_edges(verts, faces, view_axis=v_axis)
        
        all_pts = []
        for p1, p2 in lines:
            ax_view.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=0.9)
            all_pts.extend([p1, p2])

        ax_view.set_aspect("equal")
        ax_view.set_title(v_title, fontsize=9.5, pad=3)
        ax_view.axis("off")

        # Точный охватывающий прямоугольник фактического рисунка вида
        if len(all_pts) > 0:
            pts_arr = np.array(all_pts)
            p_min = pts_arr.min(axis=0)
            p_max = pts_arr.max(axis=0)
            p_range = np.maximum(p_max - p_min, 1e-4)

            # Внутри ax_view масштаб equal, найдем положение на листе
            # С запасом под размеры и надпись вида (±5 мм)
            v_box_xmin = max(frame_xmin, vx - 2.0)
            v_box_xmax = min(W0 - 5.0, vx + vw + 2.0)
            v_box_ymin = max(frame_ymin, vy - 2.0)
            v_box_ymax = min(frame_ymax, vy + vh + 4.0)
        else:
            v_box_xmin, v_box_ymin = vx, vy
            v_box_xmax, v_box_ymax = vx + vw, vy + vh

        view_boxes.append((v_name, v_box_xmin, v_box_ymin, v_box_xmax, v_box_ymax))

    # Общее поле всех видов (views_field)
    all_v_xmin = min(b[1] for b in view_boxes)
    all_v_ymin = min(b[2] for b in view_boxes)
    all_v_xmax = max(b[3] for b in view_boxes)
    all_v_ymax = max(b[4] for b in view_boxes)

    # Собираем список всех элементов для разметки YOLO:
    # (class_id, xmin, ymin, xmax, ymax)
    targets = [
        (0, all_v_xmin, all_v_ymin, all_v_xmax, all_v_ymax),                # views_field
        (1, view_boxes[0][1], view_boxes[0][2], view_boxes[0][3], view_boxes[0][4]), # view_front
        (2, view_boxes[1][1], view_boxes[1][2], view_boxes[1][3], view_boxes[1][4]), # view_top
        (3, view_boxes[2][1], view_boxes[2][2], view_boxes[2][3], view_boxes[2][4]), # view_left
        (4, stamp_xmin, stamp_ymin, stamp_xmax, stamp_ymax),                # stamp
        (5, tt_xmin, tt_ymin, tt_xmax, tt_ymax),                            # tech_req
        (6, extra_xmin, extra_ymin, extra_xmax, extra_ymax),                # stamp_extra
    ]

    # Сохраняем чертеж как изображение
    os.makedirs(os.path.dirname(os.path.abspath(out_img_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_txt_path)), exist_ok=True)

    fig.savefig(out_img_path, format="png", dpi=dpi, bbox_inches=None)
    plt.close(fig)

    # Формируем и сохраняем файл разметки YOLO (.txt)
    yolo_lines = []
    for cls_id, xmin, ymin, xmax, ymax in targets:
        xc, yc, w, h = to_yolo_bbox(xmin, ymin, xmax, ymax, W0, H0)
        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) + "\n")

    # Опционально: создаем превью с нарисованными рамками для быстрой визуальной проверки
    if preview_path:
        os.makedirs(os.path.dirname(os.path.abspath(preview_path)), exist_ok=True)
        with Image.open(out_img_path) as im:
            im_w, im_h = im.size
            prev_fig, prev_ax = plt.subplots(figsize=(10, 7), dpi=100)
            prev_ax.imshow(im)
            prev_ax.axis("off")

            colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
            for cls_id, xmin, ymin, xmax, ymax in targets:
                xc, yc, w, h = to_yolo_bbox(xmin, ymin, xmax, ymax, W0, H0)
                px_x = (xc - w / 2.0) * im_w
                px_y = (yc - h / 2.0) * im_h
                px_w = w * im_w
                px_h = h * im_h

                col = colors[cls_id % len(colors)]
                rect_p = patches.Rectangle((px_x, px_y), px_w, px_h, fill=False, edgecolor=col, lw=2)
                prev_ax.add_patch(rect_p)
                prev_ax.text(
                    px_x + 3, px_y + 12,
                    CLASSES_NAMES[cls_id],
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor=col, edgecolor="none", alpha=0.85)
                )

            prev_fig.savefig(preview_path, bbox_inches="tight", dpi=100)
            plt.close(prev_fig)

    return True

def create_dataset_yaml(out_dir):
    yaml_content = f"""# Конфигурация датасета для YOLO (ЕСКД анализ чертежей БВК / ЮУрГУ)
path: {os.path.abspath(out_dir).replace('\\', '/')}
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
"""
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path

def worker_task(task):
    obj_p, img_p, txt_p, prev_p, fmt = task
    try:
        ok = render_and_label_sheet(obj_p, img_p, txt_p, fmt=fmt, preview_path=prev_p)
        return ok
    except Exception as e:
        return False

def main():
    parser = argparse.ArgumentParser(description="Генератор синтетических чертежей ЕСКД с разметкой для YOLO")
    parser.add_argument(
        "--src",
        default=r"C:\Users\Yaroslav\OneDrive\Desktop\датасет проекта\Скрипт пайтон 3D =_ 2D и тестовый датасет\dataset_org_norm",
        help="Путь к исходным 3D-моделям (.obj)"
    )
    parser.add_argument(
        "--out",
        default="dataset_yolo",
        help="Папка для сохранения датасета YOLO"
    )
    parser.add_argument("--count", type=int, default=100, help="Количество чертежей для генерации (None = все)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Доля валидационной выборки (0.15 = 15%)")
    parser.add_argument("--format", default="A3", choices=["A3", "A4"], help="Формат листа")
    parser.add_argument("--workers", type=int, default=None, help="Количество ядер CPU")
    parser.add_argument("--preview_count", type=int, default=5, help="Количество превью-картинок с отрисованными боксами")
    args = parser.parse_args()

    print(f"=== ГЕНЕРАТОР СИНТЕТИКИ ДЛЯ YOLO (ЕСКД ЧЕРТЕЖИ) ===")
    print(f"Исходная папка: {args.src}")
    print(f"Выходная папка: {args.out}")

    all_objs = sorted(glob.glob(os.path.join(args.src, "**", "*.obj"), recursive=True))
    if not all_objs:
        print(f"[ОШИБКА] Не найдено файлов .obj по пути {args.src}")
        return

    print(f"Найдено 3D-моделей: {len(all_objs)}")
    random.seed(42)
    random.shuffle(all_objs)

    total_to_gen = min(len(all_objs), args.count) if args.count else len(all_objs)
    selected_objs = all_objs[:total_to_gen]

    # Делим на train / val
    n_val = max(1, int(total_to_gen * args.val_ratio))
    val_set = set(selected_objs[:n_val])
    train_set = set(selected_objs[n_val:])

    print(f"Будет создано: {total_to_gen} (Train: {len(train_set)}, Val: {len(val_set)})")

    # Создаем структуру папок YOLO
    os.makedirs(os.path.join(args.out, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "preview_labels"), exist_ok=True)

    tasks = []
    for idx, obj_p in enumerate(selected_objs):
        split = "val" if obj_p in val_set else "train"
        base = os.path.splitext(os.path.basename(obj_p))[0]
        cat = os.path.basename(os.path.dirname(obj_p)).replace(" ", "_").replace(",", "")
        sample_name = f"{cat}_{base}"

        img_p = os.path.join(args.out, "images", split, f"{sample_name}.png")
        txt_p = os.path.join(args.out, "labels", split, f"{sample_name}.txt")

        prev_p = None
        if idx < args.preview_count:
            prev_p = os.path.join(args.out, "preview_labels", f"{sample_name}_bboxes.png")

        tasks.append((obj_p, img_p, txt_p, prev_p, args.format))

    workers = args.workers or os.cpu_count() or 4
    print(f"Запуск генерации на {workers} ядрах CPU...")

    t0 = time.time()
    success = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_task, t): t for t in tasks}
        for done_idx, f in enumerate(concurrent.futures.as_completed(futures), 1):
            if f.result():
                success += 1
            if done_idx % 10 == 0 or done_idx == total_to_gen:
                pct = (done_idx / total_to_gen) * 100
                speed = done_idx / max(0.001, time.time() - t0)
                sys.stdout.write(f"\r  Прогресс: {done_idx}/{total_to_gen} ({pct:.1f}%) | Успешно: {success} | {speed:.1f} черт./с")
                sys.stdout.flush()

    print()
    yaml_path = create_dataset_yaml(args.out)
    print(f"\n[ГОТОВО] Сгенерировано {success} образцов за {time.time() - t0:.1f} сек.")
    print(f"Конфиг для обучения YOLO: {yaml_path}")
    print(f"Проверочные превью с боксами: {os.path.join(args.out, 'preview_labels')}")

if __name__ == "__main__":
    main()
