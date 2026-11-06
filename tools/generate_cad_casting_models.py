# -*- coding: utf-8 -*-
"""
Процедурный генератор 3D-моделей литых промышленных изделий (.obj).
Проект: Оценка себестоимости уникального литья (ЮУрГУ & ООО «БВК» / ПГ КОНАР).

Генерирует 8 классов классификатора ЕСКД:
  01: Фланцы и диски (воротниковые, плоские, упорные диски)
  02: Колеса и шкивы (шкивы клиновые с ручьями, рабочие колеса насосов, зубчатые венцы)
  03: Втулки и стаканы (направляющие с фланцем, гильзы цилиндров)
  04: Валы и оси (ступенчатые валы, оси балансиров)
  05: Корпусные детали и арматура (корпуса задвижек с патрубками, улитки насосов, корпуса редукторов)
  06: Кронштейны и опоры (силовые кронштейны с проушинами, опоры подшипников)
  07: Рычаги и крюки (двуплечие рычаги, грузовые крюки)
  08: Опорные фасонные плиты (плиты фундамента турбины с ребрами)
"""

import os
import math
import numpy as np
import trimesh

OUT_DIR = "models_cad_casting"

def make_cylinder(radius, height, center=(0, 0, 0), axis=(0, 0, 1), sections=48):
    c = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    c.apply_translation(center)
    return c

def make_box(extents, center=(0, 0, 0)):
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b

def make_annulus(r_inner, r_outer, height, center=(0, 0, 0), sections=48):
    c_out = trimesh.creation.cylinder(radius=r_outer, height=height, sections=sections)
    c_out.apply_translation(center)
    return c_out

# ----------------- 01: ФЛАНЦЫ И ДИСКИ -----------------
def generate_flange(id_num, r_bore=25, r_outer=90, r_neck=45, flange_thick=16, neck_h=35, num_holes=8):
    parts = []
    # 1. Основной диск фланца
    disc = make_cylinder(r_outer, flange_thick, center=(0, 0, flange_thick / 2.0))
    parts.append(disc)

    # 2. Воротниковая шейка под приварку
    neck = make_cylinder(r_neck, neck_h, center=(0, 0, flange_thick + neck_h / 2.0))
    parts.append(neck)

    # 3. Уплотнительное кольцо зеркала фланца (снизу)
    mirror = make_cylinder(r_neck * 1.15, 4.0, center=(0, 0, -2.0))
    parts.append(mirror)

    # 4. Ступица крепления
    pcd = (r_outer + r_neck) / 2.0
    for i in range(num_holes):
        ang = 2.0 * math.pi * i / num_holes
        hx = pcd * math.cos(ang)
        hy = pcd * math.sin(ang)
        # Болтовая бобышка
        boss = make_cylinder(r_outer * 0.08, flange_thick * 1.05, center=(hx, hy, flange_thick / 2.0))
        parts.append(boss)

    mesh = trimesh.util.concatenate(parts)
    return mesh

# ----------------- 02: КОЛЁСА И ШКИВЫ -----------------
def generate_pulley(id_num, r_hub=30, r_rim=110, width=50, num_grooves=2):
    parts = []
    # 1. Ступица со шпоночным пазом
    hub = make_cylinder(r_hub, width * 1.1, center=(0, 0, 0))
    parts.append(hub)

    # 2. Диск шкива
    disc = make_cylinder(r_rim * 0.85, width * 0.35, center=(0, 0, 0))
    parts.append(disc)

    # 3. Обод шкива с клиновыми ручьями
    rim = make_cylinder(r_rim, width, center=(0, 0, 0))
    parts.append(rim)

    # Окна облегчения в диске
    num_spokes = 4
    spoke_r = (r_hub + r_rim * 0.85) / 2.0
    for i in range(num_spokes):
        ang = 2.0 * math.pi * i / num_spokes
        sx = spoke_r * math.cos(ang)
        sy = spoke_r * math.sin(ang)
        hole_boss = make_cylinder(r_rim * 0.14, width * 0.45, center=(sx, sy, 0))
        parts.append(hole_boss)

    mesh = trimesh.util.concatenate(parts)
    return mesh

def generate_impeller(id_num, r_hub=25, r_outer=100, width=40, num_blades=6):
    parts = []
    # Коренной диск
    shroud_back = make_cylinder(r_outer, 8.0, center=(0, 0, 4.0))
    parts.append(shroud_back)

    # Ступица
    hub = make_cylinder(r_hub, width, center=(0, 0, width / 2.0))
    parts.append(hub)

    # Покрывной диск с центральным входом
    shroud_front = make_cylinder(r_outer, 6.0, center=(0, 0, width - 3.0))
    parts.append(shroud_front)

    # Лопатки двоякой кривизны
    for i in range(num_blades):
        ang0 = 2.0 * math.pi * i / num_blades
        for step in range(6):
            r_curr = r_hub + (r_outer - r_hub) * (step / 5.0)
            ang_curr = ang0 + (step * 0.25)
            bx = r_curr * math.cos(ang_curr)
            by = r_curr * math.sin(ang_curr)
            blade_seg = make_cylinder(5.0, width - 10.0, center=(bx, by, width / 2.0))
            parts.append(blade_seg)

    mesh = trimesh.util.concatenate(parts)
    return mesh

# ----------------- 03: ВТУЛКИ И СТАКАНЫ -----------------
def generate_bushing(id_num, r_body=35, r_flange=60, length=90, flange_h=15):
    parts = []
    # Тело втулки
    body = make_cylinder(r_body, length, center=(0, 0, length / 2.0))
    parts.append(body)

    # Упорный фланец
    flange = make_cylinder(r_flange, flange_h, center=(0, 0, flange_h / 2.0))
    parts.append(flange)

    # Масляный поясок (кольцевая проточка)
    groove_boss = make_cylinder(r_body * 1.08, 12.0, center=(0, 0, length * 0.6))
    parts.append(groove_boss)

    mesh = trimesh.util.concatenate(parts)
    return mesh

# ----------------- 04: ВАЛЫ И ОСИ -----------------
def generate_stepped_shaft(id_num, total_length=220):
    parts = []
    # Шейка 1 (под подшипник)
    s1 = make_cylinder(25, 45, center=(0, 0, 22.5))
    parts.append(s1)

    # Шейка 2 (бочка под зубчатый венец)
    s2 = make_cylinder(42, 90, center=(0, 0, 45 + 45))
    parts.append(s2)

    # Бурт упорный
    collar = make_cylinder(50, 15, center=(0, 0, 135 + 7.5))
    parts.append(collar)

    # Шейка 3 (вторая опора)
    s3 = make_cylinder(30, 70, center=(0, 0, 150 + 35))
    parts.append(s3)

    mesh = trimesh.util.concatenate(parts)
    return mesh

# ----------------- 05: КОРПУСА И АРМАТУРА -----------------
def generate_valve_body(id_num, body_r=65, nozzle_r=38, flange_r=70, length=180, height=130):
    parts = []
    # 1. Центральная сферическая камера задвижки
    center_sphere = trimesh.creation.icosphere(subdivisions=3, radius=body_r)
    parts.append(center_sphere)

    # 2. Левый патрубок с фланцем
    p1 = make_cylinder(nozzle_r, length / 2.0, center=(-length / 4.0, 0, 0), axis=(1, 0, 0))
    p1.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    p1.apply_translation([-length / 4.0, 0, 0])
    parts.append(p1)

    f1 = make_cylinder(flange_r, 18.0)
    f1.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    f1.apply_translation([-length / 2.0, 0, 0])
    parts.append(f1)

    # 3. Правый патрубок с фланцем
    p2 = make_cylinder(nozzle_r, length / 2.0)
    p2.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    p2.apply_translation([length / 4.0, 0, 0])
    parts.append(p2)

    f2 = make_cylinder(flange_r, 18.0)
    f2.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    f2.apply_translation([length / 2.0, 0, 0])
    parts.append(f2)

    # 4. Верхняя горловина бугеля (под шток)
    neck = make_cylinder(nozzle_r * 1.1, height / 2.0, center=(0, 0, height / 4.0))
    parts.append(neck)

    top_flange = make_cylinder(flange_r * 0.9, 16.0, center=(0, 0, height / 2.0))
    parts.append(top_flange)

    mesh = trimesh.util.concatenate(parts)
    return mesh

# ----------------- 06: КРОНШТЕЙНЫ И ОПОРЫ -----------------
def generate_bracket(id_num, base_w=140, base_d=90, base_h=16, boss_r=35, boss_len=60, height=85):
    parts = []
    # 1. Опорная плита основания
    base = make_box([base_w, base_d, base_h], center=(0, 0, base_h / 2.0))
    parts.append(base)

    # 2. Стойка с проушиной
    upright = make_box([base_w * 0.45, base_h * 1.4, height], center=(0, 0, height / 2.0))
    parts.append(upright)

    # 3. Бобышка проушины под палец
    boss = make_cylinder(boss_r, boss_len)
    boss.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    boss.apply_translation([0, 0, height])
    parts.append(boss)

    # 4. Два диагональных ребра жесткости
    rib1 = make_box([base_h * 0.9, base_d * 0.4, height * 0.6], center=(-base_w * 0.3, 0, height * 0.35))
    rib2 = make_box([base_h * 0.9, base_d * 0.4, height * 0.6], center=(base_w * 0.3, 0, height * 0.35))
    parts.append(rib1)
    parts.append(rib2)

    mesh = trimesh.util.concatenate(parts)
    return mesh

# ----------------- 07: РЫЧАГИ И КРЮКИ -----------------
def generate_lever(id_num, arm_len=160, r_large=38, r_small=24, thick=20):
    parts = []
    # Большая ступица (ось качания)
    h_large = make_cylinder(r_large, thick * 1.5, center=(0, 0, 0))
    parts.append(h_large)

    # Малая бобышка (тяга)
    h_small = make_cylinder(r_small, thick * 1.2, center=(arm_len, 0, 0))
    parts.append(h_small)

    # Тело рычага (балка переменного сечения)
    beam = make_box([arm_len, (r_large + r_small) * 0.7, thick], center=(arm_len / 2.0, 0, 0))
    parts.append(beam)

    mesh = trimesh.util.concatenate(parts)
    return mesh

# ----------------- 08: ОПОРНЫЕ ПЛИТЫ -----------------
def generate_base_plate(id_num, width=200, depth=140, thick=22, rib_h=40):
    parts = []
    # Основная плита
    plate = make_box([width, depth, thick], center=(0, 0, thick / 2.0))
    parts.append(plate)

    # Продольные ребра жесткости
    r1 = make_box([width * 0.9, thick * 0.7, rib_h], center=(0, depth * 0.3, thick + rib_h / 2.0))
    r2 = make_box([width * 0.9, thick * 0.7, rib_h], center=(0, -depth * 0.3, thick + rib_h / 2.0))
    parts.append(r1)
    parts.append(r2)

    # Центральная посадочная площадка под агрегат
    pad = make_box([width * 0.45, depth * 0.5, thick * 0.8], center=(0, 0, thick + thick * 0.4))
    parts.append(pad)

    mesh = trimesh.util.concatenate(parts)
    return mesh

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Генерация параметрических 3D CAD моделей литых изделий (.obj) ===")
    
    generators = [
        ("01_Flange", generate_flange, 6),
        ("02_Pulley", generate_pulley, 4),
        ("02_Impeller", generate_impeller, 4),
        ("03_Bushing", generate_bushing, 6),
        ("04_Shaft", generate_stepped_shaft, 5),
        ("05_Valve", generate_valve_body, 6),
        ("06_Bracket", generate_bracket, 6),
        ("07_Lever", generate_lever, 5),
        ("08_BasePlate", generate_base_plate, 5),
    ]

    count = 0
    for name_prefix, gen_func, num_variants in generators:
        class_folder = os.path.join(OUT_DIR, name_prefix)
        os.makedirs(class_folder, exist_ok=True)

        for v in range(1, num_variants + 1):
            # Вариации масштаба для разнообразия геометрии
            scale_factor = 0.85 + 0.1 * v
            mesh = gen_func(v)
            mesh.apply_scale(scale_factor)

            obj_filename = f"{name_prefix}_v{v:02d}.obj"
            out_path = os.path.join(class_folder, obj_filename)
            mesh.export(out_path)
            count += 1
            print(f"  [{count:02d}] Создан 3D-меш: {out_path} ({len(mesh.vertices)} верш., {len(mesh.faces)} граней)")

    print("=" * 60)
    print(f"[УСПЕХ] Сгенерировано {count} высококачественных 3D CAD моделей в папке '{OUT_DIR}'!")

if __name__ == "__main__":
    main()
