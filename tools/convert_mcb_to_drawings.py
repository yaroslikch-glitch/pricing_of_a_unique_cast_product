# -*- coding: utf-8 -*-
"""
Пакетный генератор 2D чертежей (PDF по ГОСТ 2.104) из 3D-моделей (.obj) датасета MCB.
Проект: Рекомендательный сервис для оценки стоимости изготовления уникальных литых изделий (ЮУрГУ / ООО "БВК").
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Справочные данные для генерации атрибутов отливки (ООО "БВК" / ПГ КОНАР)
STEEL_GRADES = [
    "Сталь 20ГЛ ГОСТ 977-88",
    "Сталь 20ХМЛ ГОСТ 977-88",
    "Сталь 12Х18Н12М3ТЛ ГОСТ 977-88",
    "Сталь 08Х18Н10ТЛ ГОСТ 977-88",
    "Сталь 20Л ГОСТ 977-88",
    "Сталь 15Х1М1ФЛ ГОСТ 977-88",
    "Сталь 10Х17Н13М2ТЛ ГОСТ 977-88",
]

CASTING_GROUPS = ["группы 2", "группы 3", "группы 3а"]

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
    "Гидравлические испытания на прочность и плотность давлением Pпр = 4.0 МПа по ГОСТ 356-80.",
    "Гидравлические испытания давлением Pпр = 6.4 МПа. Выдержка не менее 10 минут.",
    "Пневматические испытания на герметичность воздухом P = 0.6 МПа под водой.",
]

# Категории MCB, наиболее релевантные для литейного завода (ООО "БВК")
FOUNDRY_CATEGORIES = [
    "Valve",
    "Elbow fitting",
    "T-shape fitting",
    "Standard fitting",
    "Impeller",
    "Turbine",
    "Flanged block bearing",
    "Flanged plain bearings",
    "Bushes",
    "Nozzle",
    "Plugs",
    "Articulations, eyelets and other articulated joints",
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

    # Устранение дубликатов граней (встречается при экспорте double-sided сеток)
    seen = set()
    dedup_faces = []
    for f in raw_faces:
        sf = tuple(sorted(f))
        if sf not in seen:
            seen.add(sf)
            dedup_faces.append(f)

    return np.array(verts), np.array(dedup_faces)


def extract_projected_edges(verts, faces, view_axis=2):
    """
    Извлечение контурных силуэтов и ребер перегиба (crease edges) для проекции.
    view_axis: 0 (X - вид слева), 1 (Y - главный вид спереди), 2 (Z - вид сверху)
    """
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
            if e not in edge_faces:
                edge_faces[e] = []
            edge_faces[e].append(f_idx)

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


def generate_drawing_pdf(
    obj_path,
    out_pdf_path,
    part_name="Корпус отливки",
    part_code="БВК.713.042.001",
    category="Valve",
    export_png=True,
):
    try:
        verts, faces = load_obj(obj_path)
    except Exception as e:
        print(f"[ERR] Ошибка чтения {obj_path}: {e}")
        return False

    if len(verts) == 0 or len(faces) == 0:
        return False

    # Формат листа А3: 420 x 297 мм
    fig = plt.figure(figsize=(16.54, 11.69), dpi=150)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, 420)
    ax_bg.set_ylim(0, 297)
    ax_bg.axis("off")

    # Внешняя рамка (ГОСТ 2.104: 20 мм слева, 5 мм с остальных)
    border = patches.Rectangle(
        (20, 5), 420 - 25, 297 - 10, linewidth=1.5, edgecolor="black", facecolor="white"
    )
    ax_bg.add_patch(border)

    # Основная надпись (Штамп ГОСТ 2.104: 185 x 55 мм)
    sx = 420 - 5 - 185
    sy = 5
    stamp = patches.Rectangle(
        (sx, sy), 185, 55, linewidth=1.2, edgecolor="black", facecolor="none"
    )
    ax_bg.add_patch(stamp)

    # Разделительные линии штампа
    ax_bg.plot([sx, sx + 185], [sy + 15, sy + 15], "k-", lw=0.8)
    ax_bg.plot([sx, sx + 185], [sy + 40, sy + 40], "k-", lw=0.8)
    ax_bg.plot([sx + 65, sx + 65], [sy, sy + 40], "k-", lw=0.8)
    ax_bg.plot([sx + 135, sx + 135], [sy, sy + 15], "k-", lw=0.8)

    # Текст штампа
    material = random.choice(STEEL_GRADES)
    casting_grp = random.choice(CASTING_GROUPS)

    ax_bg.text(
        sx + 68,
        sy + 44,
        part_code,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        family="sans-serif",
    )
    ax_bg.text(
        sx + 68,
        sy + 26,
        part_name,
        fontsize=12,
        fontweight="bold",
        va="center",
        family="sans-serif",
    )
    ax_bg.text(
        sx + 68, sy + 7, material, fontsize=9.5, va="center", family="sans-serif"
    )
    ax_bg.text(
        sx + 140,
        sy + 7,
        "ООО «БВК»",
        fontsize=10.5,
        fontweight="bold",
        va="center",
        family="sans-serif",
    )

    # Фамилии разработчиков из команды ЮУрГУ
    ax_bg.text(sx + 5, sy + 48, "Разраб.   Салов В.К.", fontsize=8)
    ax_bg.text(sx + 5, sy + 38, "Пров.     Янчук А.П.", fontsize=8)
    ax_bg.text(sx + 5, sy + 28, "Т.контр.  Симонова С.С.", fontsize=8)
    ax_bg.text(sx + 5, sy + 18, "Н.контр.  Тупикин М.Т.", fontsize=8)
    ax_bg.text(sx + 5, sy + 8, "Утв.      Иванова О.Н.", fontsize=8)

    # Технические требования (ТТ)
    selected_ndt = random.sample(NDT_REQUIREMENTS, k=random.randint(2, 4))
    ht = random.choice(HEAT_TREATMENTS)
    pt = random.choice(PRESSURE_TESTS)

    tt_lines = [
        "Технические требования:",
        "1. * Размеры для справок.",
        f"2. Отливка {casting_grp} по ГОСТ 977-88.",
        "3. Неуказанные литейные радиусы R3...R5 мм.",
        "4. Контроль неразрушающими методами:",
    ]
    for ndt_item in selected_ndt:
        tt_lines.append(f"   - {ndt_item};")
    tt_lines.extend(
        [
            f"5. Термообработка: {ht}",
            f"6. {pt}",
            "7. Формовочные уклоны по ГОСТ 3212-92.",
            "8. Маркировать клеймением: номер плавки, марка стали, шифр изделия.",
        ]
    )

    tt_text = "\n".join(tt_lines)
    ax_bg.text(
        sx - 5,
        sy + 60,
        tt_text,
        fontsize=8.5,
        va="bottom",
        ha="left",
        family="sans-serif",
        linespacing=1.3,
    )

    # Ортогональные проекции
    views = [(1, "Главный вид"), (2, "Вид сверху"), (0, "Вид слева")]
    rects = [
        [0.08, 0.44, 0.35, 0.45],  # Front
        [0.08, 0.08, 0.35, 0.32],  # Top
        [0.46, 0.44, 0.35, 0.45],  # Left
    ]

    for (v_ax, v_title), rect in zip(views, rects):
        ax_view = fig.add_axes(rect)
        lines = extract_projected_edges(verts, faces, view_axis=v_ax)
        for p1, p2 in lines:
            ax_view.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=0.85)
        ax_view.set_aspect("equal")
        ax_view.set_title(v_title, fontsize=11, pad=5)
        ax_view.axis("off")

    os.makedirs(os.path.dirname(os.path.abspath(out_pdf_path)), exist_ok=True)
    plt.savefig(out_pdf_path, format="pdf", bbox_inches="tight")
    if export_png:
        plt.savefig(
            out_pdf_path.replace(".pdf", ".png"),
            format="png",
            dpi=130,
            bbox_inches="tight",
        )
    plt.close()
    return True


def _worker_task(args):
    obj_file, out_pdf, part_name, part_code, cat, export_png, skip_existing = args
    if skip_existing and os.path.exists(out_pdf):
        if not export_png or os.path.exists(out_pdf.replace(".pdf", ".png")):
            return False, "skipped"
    ok = generate_drawing_pdf(
        obj_file,
        out_pdf,
        part_name=part_name,
        part_code=part_code,
        category=cat,
        export_png=export_png,
    )
    return ok, "generated" if ok else "error"


def batch_convert(
    dataset_root,
    output_dir,
    categories=None,
    limit_per_category=0,
    export_png=True,
    all_categories=False,
    skip_existing=True,
    workers=4,
):
    import concurrent.futures

    if all_categories:
        train_dir = os.path.join(dataset_root, "train")
        if os.path.exists(train_dir):
            categories = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        else:
            categories = FOUNDRY_CATEGORIES
    elif categories is None:
        categories = FOUNDRY_CATEGORIES

    print(f"Генерация чертежей из: {dataset_root}")
    print(f"Каталог сохранения:    {output_dir}")
    print(f"Выбрано категорий:     {len(categories)}")
    limit_str = "ВСЕ модели (без лимита)" if limit_per_category <= 0 else f"{limit_per_category} на категорию"
    print(f"Лимит на категорию:    {limit_str}")
    print(f"Количество потоков:    {workers}")
    print(f"Пропуск существующих:  {'Да' if skip_existing else 'Нет'}")
    print("-" * 60)

    tasks = []
    for split in ["train", "test"]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            continue

        for cat in categories:
            cat_path = os.path.join(split_dir, cat)
            if not os.path.exists(cat_path):
                continue

            obj_files = glob.glob(os.path.join(cat_path, "*.obj"))
            count = len(obj_files) if limit_per_category <= 0 else min(len(obj_files), limit_per_category)

            for i in range(count):
                obj_file = obj_files[i]
                basename = os.path.splitext(os.path.basename(obj_file))[0]
                out_pdf = os.path.join(output_dir, split, cat, f"{basename}_drawing.pdf")
                part_code = f"БВК.{random.randint(700, 799)}.{random.randint(100, 999)}.{basename[-3:]}"
                part_name = f"Отливка ({cat})"
                tasks.append((obj_file, out_pdf, part_name, part_code, cat, export_png, skip_existing))

    total_tasks = len(tasks)
    print(f"Всего задач к выполнению: {total_tasks}")

    total_generated = 0
    total_skipped = 0

    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, (ok, status) in enumerate(executor.map(_worker_task, tasks), 1):
                if status == "generated":
                    total_generated += 1
                elif status == "skipped":
                    total_skipped += 1
                if idx % 10 == 0 or idx == total_tasks:
                    sys.stdout.write(f"\r  Прогресс: {idx}/{total_tasks} | Создано: {total_generated} | Пропущено (уже есть): {total_skipped}")
                    sys.stdout.flush()
    else:
        for idx, task in enumerate(tasks, 1):
            ok, status = _worker_task(task)
            if status == "generated":
                total_generated += 1
            elif status == "skipped":
                total_skipped += 1
            if idx % 5 == 0 or idx == total_tasks:
                sys.stdout.write(f"\r  Прогресс: {idx}/{total_tasks} | Создано: {total_generated} | Пропущено (уже есть): {total_skipped}")
                sys.stdout.flush()

    print(f"\n\n[ГОТОВО] Сгенерировано новых чертежей: {total_generated}, ранее существовало: {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерация 2D-чертежей ГОСТ из 3D OBJ датасета MCB"
    )
    parser.add_argument(
        "--input",
        default=r"f:\Трек\dataset_org_norm",
        help="Путь к датасету MCB",
    )
    parser.add_argument(
        "--output",
        default=r"f:\Трек\dataset_2d_drawings",
        help="Папка для сохранения чертежей PDF",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Количество чертежей на категорию (0 = все без ограничений)",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Обрабатывать все 68 категорий, а не только литейные",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Количество параллельных процессов (по умолчанию 4)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Перезаписывать уже существующие чертежи",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Путь к одному файлу .obj для тестовой конвертации",
    )
    args = parser.parse_args()

    if args.single:
        out_pdf = os.path.splitext(args.single)[0] + "_drawing.pdf"
        generate_drawing_pdf(args.single, out_pdf)
        print(f"Сгенерирован: {out_pdf}")
    else:
        batch_convert(
            args.input,
            args.output,
            limit_per_category=args.limit,
            all_categories=args.all_categories,
            skip_existing=not args.no_skip,
            workers=args.workers,
        )
