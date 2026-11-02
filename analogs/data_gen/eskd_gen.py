"""Генератор синтетических чертежей в стиле ЕСКД из 3D-моделей MCB (.obj).

Запуск:  python eskd_gen.py --src mcb_raw --out out --workers 2
Вход:    mcb_raw/<наш_класс>/*.obj
Выход:   out/pdf/<класс>/<обозн>.pdf     — полный лист (рамка, штамп, ТТ)
         out/png_views/<класс>/<обозн>.png — только поле видов, 512x512, для обучения
         out/params.csv, out/inventory.csv, out/split.csv, out/preview/<класс>.png
"""
import argparse, csv, hashlib, math, os, random, sys, traceback
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Polygon as MplPolygon, PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans", "font.style": "italic",
                     "pdf.fonttype": 42, "hatch.linewidth": 0.6})

S_LW, T_LW = 1.9, 0.6                       # основная / тонкая, pt
DASH = (0, (6, 3)); AXIS = (0, (14, 3, 2, 3))
SCALES = [(4, "4:1"), (2, "2:1"), (1, "1:1"), (0.5, "1:2"), (0.4, "1:2,5"),
          (0.25, "1:4"), (0.2, "1:5"), (0.1, "1:10")]
NAMES = {"01": ["Фланец", "Диск", "Крышка"], "02": ["Колесо", "Шкив", "Колесо рабочее"],
         "03": ["Втулка", "Стакан", "Кольцо"], "04": ["Вал", "Ось", "Палец"],
         "05": ["Корпус", "Корпус подшипника", "Корпус клапана"],
         "06": ["Кронштейн", "Опора", "Стойка"], "07": ["Рычаг", "Вилка", "Серьга"],
         "08": ["Планка", "Пластина", "Накладка"]}
MATERIALS = ["Сталь 20Л ГОСТ 977-88", "Сталь 25Л ГОСТ 977-88", "Сталь 35Л ГОСТ 977-88",
             "Сталь 20ГЛ ГОСТ 977-88", "Сталь 12Х18Н9ТЛ ГОСТ 977-88", "СЧ 20 ГОСТ 1412-85"]

# ---------------------------------------------------------------- геометрия
def load_mesh(path, max_faces=60000):
    m = trimesh.load(path, force="mesh", process=True)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) < 12:
        raise ValueError("empty_or_tiny")
    digits = max(1, int(6 - np.log10(max(m.extents.max(), 1e-9))))
    m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=digits)
    m.update_faces(m.nondegenerate_faces()); m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    if len(m.faces) > max_faces:
        m = m.simplify_quadric_decimation(face_count=max_faces)
    return m

def orient(m, rng):
    m.apply_translation(-m.bounds.mean(0))
    pts = m.sample(6000)
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)   # строки: оси по убыванию
    ext = np.ptp((pts - pts.mean(0)) @ vt.T, axis=0)
    a, b, c = ext
    revolution = False
    if (a - b) / a < 0.06:            # диск: ось вращения = короткая
        order, revolution = [2, 1, 0], True
    elif (b - c) / b < 0.06:          # вал: ось вращения = длинная
        order, revolution = [0, 1, 2], True
    else:                             # призматическая: длинная→X, средняя→Z, короткая→Y
        order = [0, 2, 1]
    R = vt[order]                     # новые оси X, Y, Z
    if np.linalg.det(R) < 0:
        R[1] *= -1
    T = np.eye(4); T[:3, :3] = R
    m.apply_transform(T)
    m.apply_translation(-m.bounds.mean(0))
    ext = m.extents
    if ext.min() / ext.max() < 0.01:
        raise ValueError("flat")
    m.apply_scale(rng.uniform(80, 600) / ext.max())
    return m, revolution

def volume_mm3(m):
    if m.is_watertight and m.volume > 0:
        return float(m.volume)
    pitch = m.extents.max() / 64
    return float(m.voxelized(pitch).fill().volume)

def fingerprint(m, rng):
    p = m.sample(2000); d = np.linalg.norm(p[:1000] - p[1000:], axis=1) / m.extents.max()
    h, _ = np.histogram(d, bins=32, range=(0, 1.8), density=True)
    e = np.sort(m.extents)[::-1] / m.extents.max()
    v = np.r_[e * 3, h / (np.linalg.norm(h) + 1e-9)]
    return v / np.linalg.norm(v)

# ---------------------------------------------------------------- линии видов
# вид: (оси на листе, знаки, направление «к зрителю»)
VIEWS = {"front": ((0, 2), (1, 1), np.array([0., -1., 0.])),
         "top":   ((0, 1), (1, 1), np.array([0., 0., 1.])),
         "left":  ((1, 2), (-1, 1), np.array([-1., 0., 0.]))}

def view_lines(m, view, sharp_deg=35, step=None):
    """-> (visible_segments, hidden_segments) в координатах листа (мм модели)."""
    (ia, ib), (sa, sb), d = VIEWS[view]
    adj, ang = m.face_adjacency, m.face_adjacency_angles
    n = m.face_normals
    sharp = ang > np.radians(sharp_deg)
    d0, d1 = n[adj[:, 0]] @ d, n[adj[:, 1]] @ d
    sil = (d0 * d1) < 0
    E = m.face_adjacency_edges[sharp | sil]
    from trimesh.grouping import group_rows
    es = m.edges_sorted
    bnd = es[group_rows(es, require_count=1)]
    if len(bnd):
        E = np.vstack([E, bnd])
    if not len(E):
        return np.zeros((0, 2, 2)), np.zeros((0, 2, 2))
    P0, P1 = m.vertices[E[:, 0]], m.vertices[E[:, 1]]
    # отбросить рёбра, вырожденные в проекции (параллельные взгляду)
    proj = lambda P: np.c_[P[:, ia] * sa, P[:, ib] * sb]
    plen = np.linalg.norm(proj(P1) - proj(P0), axis=1)
    keep = plen > 1e-6 * m.extents.max()
    P0, P1, plen = P0[keep], P1[keep], plen[keep]
    step = step or m.extents.max() / 150
    nseg = np.clip(np.ceil(plen / step).astype(int), 1, 60)
    idx = np.repeat(np.arange(len(P0)), nseg)
    k = np.concatenate([np.arange(c) for c in nseg])
    t0 = k / nseg[idx]; t1 = (k + 1) / nseg[idx]
    A = P0[idx] + (P1[idx] - P0[idx]) * t0[:, None]
    B = P0[idx] + (P1[idx] - P0[idx]) * t1[:, None]
    mid = (A + B) / 2
    eps = 1e-3 * m.extents.max()
    origins = mid + d * eps
    hit = m.ray.intersects_any(origins, np.tile(d, (len(mid), 1)))
    segs = np.stack([proj(A), proj(B)], axis=1)
    return segs[~hit], segs[hit]

def merge_len(segs):
    return float(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum()) if len(segs) else 0.0

def half_section(m):
    """Задняя половина детали (y>=0) с заглушкой по плоскости разреза."""
    try:
        back = m.slice_plane(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0], cap=True)
        if back is None or len(back.faces) < 10:
            return None, None
        sec = m.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])
        if sec is None:
            return None, None
        polys = []
        for ent in sec.discrete:
            if len(ent) >= 4 and np.allclose(ent[0], ent[-1], atol=1e-6 * m.extents.max()):
                polys.append(np.c_[ent[:, 0], ent[:, 2]])
        if not polys:
            return None, None
        from shapely.geometry import Polygon as SP, box
        geom = None
        for p in polys:
            g = SP(p).buffer(0)
            geom = g if geom is None else geom.symmetric_difference(g)
        ext = m.extents
        geom = geom.intersection(box(-ext[0], -ext[2], ext[0], 0))
        if geom.is_empty:
            return None, None
        return back, geom
    except Exception:
        return None, None

def clip_segs(segs, cond):
    """оставить отрезки, середина которых удовлетворяет cond(xy)."""
    if not len(segs):
        return segs
    return segs[cond((segs[:, 0] + segs[:, 1]) / 2)]

def build_views(m, revolution):
    out = {}
    for v in VIEWS:
        vis, hid = view_lines(m, v)
        if merge_len(hid) > 2.5 * merge_len(vis):       # загромождение — без невидимых
            hid = hid[:0]
        out[v] = {"vis": vis, "hid": hid, "hatch": [], "axis": []}
    sect = False
    if revolution:
        back, polys = half_section(m)
        if back is not None:
            vis_b, _ = view_lines(back, "front")
            f = out["front"]
            f["vis"] = np.vstack([clip_segs(f["vis"], lambda p: p[:, 1] >= 0),
                                  clip_segs(vis_b, lambda p: p[:, 1] < 0)])
            f["hid"] = clip_segs(f["hid"], lambda p: p[:, 1] >= 0)
            f["hatch"] = polys; f["hatch_below"] = True
            sect = True
    if revolution:
        ex = m.extents
        for v in ("front", "top"):
            out[v]["axis"].append(((-ex[0] / 2 - 4, 0), (ex[0] / 2 + 4, 0)))
        r = max(ex[1], ex[2]) / 2 + 4
        out["left"]["axis"] += [((-r, 0), (r, 0)), ((0, -r), (0, r))]
    return out, sect

# ---------------------------------------------------------------- рисование
def draw_views(ax, views, origins, s, ext, revolution, dims=True):
    """views в мм модели, origins[v] = центр вида на листе (мм), s — масштаб."""
    for v, (ox, oy) in origins.items():
        V = views[v]
        tr = lambda segs: segs * s + np.array([ox, oy])
        if len(V["vis"]):
            ax.add_collection(LineCollection(tr(V["vis"]), colors="k", linewidths=S_LW, capstyle="round"))
        if len(V["hid"]):
            ax.add_collection(LineCollection(tr(V["hid"]), colors="k", linewidths=T_LW, linestyles=DASH))
        g = V.get("hatch")
        if g is not None and not isinstance(g, list):
            for pg in getattr(g, "geoms", [g]):
                if pg.geom_type != "Polygon" or pg.area <= 0:
                    continue
                rings = [np.asarray(pg.exterior.coords)] + [np.asarray(r.coords) for r in pg.interiors]
                verts, codes = [], []
                for r in rings:
                    r = r * s + np.array([ox, oy])
                    verts += list(r); codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 2) + [MplPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, hatch="////", lw=0, edgecolor="k"))
        for (a, b) in V["axis"]:
            ax.plot([ox + a[0] * s, ox + b[0] * s], [oy + a[1] * s, oy + b[1] * s],
                    color="k", lw=T_LW, ls=AXIS)
        if V.get("hatch_below"):
            ax.plot([ox - ext[0] / 2 * s, ox + ext[0] / 2 * s], [oy, oy], color="k", lw=T_LW, ls=AXIS)
    if not dims:
        return
    L, W, H = ext
    fx, fy = origins["front"]; tx, ty = origins["top"]; lx, ly = origins["left"]
    dimh(ax, fx - L / 2 * s, fx + L / 2 * s, fy + H / 2 * s + 10, fy + H / 2 * s, f"{L:.0f}*")
    if revolution:
        dimv(ax, fx - L / 2 * s - 10, fy - H / 2 * s, fy + H / 2 * s, fx - L / 2 * s, f"Ø{H:.0f}*")
    else:
        dimv(ax, fx - L / 2 * s - 10, fy - H / 2 * s, fy + H / 2 * s, fx - L / 2 * s, f"{H:.0f}*")
        dimv(ax, tx - L / 2 * s - 10, ty - W / 2 * s, ty + W / 2 * s, tx - L / 2 * s, f"{W:.0f}*")

def arrow(ax, a, b):
    ax.annotate("", a, b, arrowprops=dict(arrowstyle="<|-|>", lw=T_LW, color="k",
                                          shrinkA=0, shrinkB=0, mutation_scale=7))

def dimh(ax, xa, xb, y, yfrom, s):
    for x in (xa, xb):
        ax.plot([x, x], [yfrom + (1 if y > yfrom else -1), y + (2 if y > yfrom else -2)], color="k", lw=T_LW)
    arrow(ax, (xa, y), (xb, y))
    ax.text((xa + xb) / 2, y + 0.8, s, fontsize=9, ha="center", va="bottom")

def dimv(ax, x, ya, yb, xfrom, s):
    for yy in (ya, yb):
        ax.plot([xfrom - 1, x - 2], [yy, yy], color="k", lw=T_LW)
    arrow(ax, (x, ya), (x, yb))
    ax.text(x - 0.8, (ya + yb) / 2, s, fontsize=9, ha="right", va="center", rotation=90)

FORMATS = {  # ширина, высота листа; область для видов (x0, y0, x1, y1); запретные зоны
    "A4": (210, 297, (25, 112, 200, 268)),
    "A3": (420, 297, (25, 10, 410, 268)),
}

def layout(ext, fmt):
    """подобрать масштаб и центры видов. -> (s, label, origins) или None"""
    W0, H0, (x0, y0, x1, y1) = FORMATS[fmt]
    L, W, H = ext
    M, G = 22, 28                         # поле под размеры, зазор между видами
    for s, lab in SCALES:
        fw, fh = L * s, H * s; lw = W * s; th = W * s
        width = M + fw + G + lw + 5
        height = M + fh + G + th + 5
        if fmt == "A3":
            # вид сверху левее штампа (x<225), вид слева выше ТТ (y>140)
            top_left = x0 + M
            if top_left + fw > 222 or y1 - M - fh - G - th < y0:
                continue
            if x0 + width > x1:
                continue
            fx = x0 + M + fw / 2; fy = y1 - fh / 2
            if fy - fh / 2 < 140:          # вид слева должен быть выше ТТ
                continue
        else:
            if width > x1 - x0 or height > y1 - y0:
                continue
            fx = x0 + M + fw / 2; fy = y1 - fh / 2
        origins = {"front": (fx, fy),
                   "left": (fx + fw / 2 + G + lw / 2, fy),
                   "top": (fx, fy - fh / 2 - G - th / 2)}
        area = (fw * fh + fw * th + lw * fh) / ((x1 - x0) * (y1 - y0))
        return s, lab, origins, area
    return None

def stamp(ax, fmt, d):
    W0, H0, _ = FORMATS[fmt]
    L = lambda xs, ys, w=S_LW: ax.plot(xs, ys, color="k", lw=w, solid_capstyle="butt")
    T = lambda x, y, s, size=9, **k: ax.text(x, y, s, fontsize=size, **k)
    ax.add_patch(Rectangle((20, 5), W0 - 25, H0 - 10, fill=False, lw=S_LW))
    ax.add_patch(Rectangle((20, H0 - 19), 70, 14, fill=False, lw=S_LW))
    T(55, H0 - 12, d["designation"], 8, ha="center", va="center", rotation=180)
    x0, y0 = W0 - 5 - 185, 5
    ax.add_patch(Rectangle((x0, y0), 185, 55, fill=False, lw=S_LW))
    for xx in [7, 17, 40, 55, 65]:
        L([x0 + xx] * 2, [y0, y0 + 55], S_LW if xx == 65 else T_LW)
    for i in range(1, 11):
        L([x0, x0 + 65], [y0 + 5 * i] * 2, S_LW if i in (5, 6) else T_LW)
    L([x0 + 65, x0 + 185], [y0 + 40] * 2); L([x0 + 65, x0 + 135], [y0 + 15] * 2)
    L([x0 + 135] * 2, [y0, y0 + 40]); L([x0 + 135, x0 + 185], [y0 + 35] * 2, T_LW)
    L([x0 + 135, x0 + 185], [y0 + 20] * 2); L([x0 + 135, x0 + 185], [y0 + 15] * 2, T_LW)
    for xx in [140, 145]:
        L([x0 + xx] * 2, [y0 + 20, y0 + 35], T_LW)
    for xx in [150, 167]:
        L([x0 + xx] * 2, [y0 + 20, y0 + 40])
    L([x0 + 155] * 2, [y0 + 15, y0 + 20], T_LW)
    for i, s in enumerate(["Изм.", "Лист", "№ докум.", "Подп.", "Дата"]):
        T(x0 + [1, 8, 19, 42, 56][i], y0 + 31.5, s, 6)
    for i, s in enumerate(["Разраб.", "Пров.", "Т.контр.", "", "Н.контр.", "Утв."]):
        T(x0 + 1, y0 + 26.5 - 5 * i, s, 6)
    T(x0 + 125, y0 + 47.5, d["designation"], 15, ha="center", va="center")
    T(x0 + 100, y0 + 27.5, d["name"], 14 if len(d["name"]) < 14 else 11, ha="center", va="center")
    T(x0 + 100, y0 + 7.5, d["material"], 9, ha="center", va="center")
    T(x0 + 142.5, y0 + 36.3, "Лит.", 6, ha="center"); T(x0 + 158.5, y0 + 36.3, "Масса", 6, ha="center")
    T(x0 + 176, y0 + 36.3, "Масштаб", 6, ha="center")
    T(x0 + 158.5, y0 + 27, d["mass_str"], 10, ha="center", va="center")
    T(x0 + 176, y0 + 27, d["scale"], 10, ha="center", va="center")
    T(x0 + 137, y0 + 16.3, "Лист 1", 6); T(x0 + 157, y0 + 16.3, "Листов 1", 6)
    T(x0 + 160, y0 + 7.5, "Учебный проект", 9, ha="center", va="center")
    # шероховатость
    T(W0 - 42, H0 - 14, d["rough"], 10)
    L([W0 - 29, W0 - 26, W0 - 22], [H0 - 12.5, H0 - 16, H0 - 9], T_LW)
    T(W0 - 20, H0 - 14, "(√)", 10)
    # технические требования
    yt = 60 + 8 + 6 * len(d["tt"])
    T(x0, yt + 2, "Технические требования", 10)
    for i, s in enumerate(d["tt"]):
        T(x0, yt - 5 - 6 * i, s, 8.5)

def tech_req(rng):
    pool = [
        "*Размеры для справок.",
        f"Отливка группы {rng.choice(['1', '2', '3'])} ГОСТ 977-88.",
        f"Точность отливки {rng.choice(['8', '9', '10'])}-{rng.choice(['4', '5'])}-"
        f"{rng.choice(['3', '4'])}-{rng.choice(['8', '9'])} См {rng.choice(['1,0', '1,6'])} ГОСТ Р 53464-2009.",
        f"Неуказанные литейные радиусы R{rng.choice(['3', '5'])}...R{rng.choice(['5', '8'])} мм.",
        "Формовочные уклоны по ГОСТ 3212-92.",
        f"Термообработка: {rng.choice(['нормализация + отпуск', 'отжиг'])}, "
        f"{rng.choice(['140...180', '160...200'])} HB.",
        "Контроль: ВИК 100%" + rng.choice([", УЗК.", ", капиллярный контроль.", "."]),
        "Маркировать номер плавки и шифр изделия.",
    ]
    k = rng.randint(5, 8)
    chosen = [pool[0]] + sorted(rng.sample(pool[1:], k - 1), key=pool.index)
    return [f"{i + 1}. {s}" for i, s in enumerate(chosen)]

# ---------------------------------------------------------------- один файл
def process(job):
    src, cls, idx, out, seed = job
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    rec = {"src_obj": src, "our_class": cls, "status": "error", "reason": ""}
    try:
        m = load_mesh(src)
        m, rev = orient(m, nrng)
        ext = m.extents
        mass = volume_mm3(m) * 7.85e-6
        fp = fingerprint(m, nrng)
        views, sect = build_views(m, rev)
        L_, W_, H_ = ext
        if merge_len(views["front"]["vis"]) < 1.5 * (L_ + H_):
            raise ValueError("sparse_lines")
        fmt_best = None
        for fmt in ("A4", "A3"):
            r = layout(ext, fmt)
            if r and (fmt_best is None or r[0] > fmt_best[1][0]):
                fmt_best = (fmt, r)
        if fmt_best is None:
            raise ValueError("no_layout")
        fmt, (s, slabel, origins, area) = fmt_best
        code = cls[:2]
        des = f"SYN.{code}.{idx:06d}"
        d = dict(designation=des, name=rng.choice(NAMES[code]), material=rng.choice(MATERIALS),
                 mass_str=f"{mass:.1f}".replace(".", ","), scale=slabel,
                 rough=f"Rz {rng.choice(['40', '80'])}", tt=tech_req(rng))
        W0, H0, _ = FORMATS[fmt]
        # полный лист
        fig = plt.figure(figsize=(W0 / 25.4, H0 / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W0); ax.set_ylim(0, H0); ax.set_aspect("equal"); ax.axis("off")
        draw_views(ax, views, origins, s, ext, rev)
        stamp(ax, fmt, d)
        pdf = Path(out, "pdf", cls, des + ".pdf"); pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf)
        if os.environ.get("GEN_FULLPNG"):
            fp_ = Path(out, "png_full", cls, des + ".png"); fp_.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fp_, dpi=int(os.environ["GEN_FULLPNG"]))
        plt.close(fig)
        # только поле видов — для обучения
        fig = plt.figure(figsize=(W0 / 25.4, H0 / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W0); ax.set_ylim(0, H0); ax.set_aspect("equal"); ax.axis("off")
        draw_views(ax, views, origins, s, ext, rev)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(-1)
        plt.close(fig)
        ys, xs = np.where(img < 200)
        crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = crop.shape; side = int(max(h, w) * 1.1)
        canvas = np.full((side, side), 255, np.uint8)
        canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop.astype(np.uint8)
        png = Path(out, "png_views", cls, des + ".png"); png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).resize((512, 512), Image.LANCZOS).save(png)
        L, W, H = ext
        rec.update(status="ok", designation=des, file_pdf=str(pdf.relative_to(out)),
                   file_view=str(png.relative_to(out)), name=d["name"], material=d["material"],
                   mass_kg=round(mass, 1), scale=slabel, format=fmt, L=round(L), W=round(W), H=round(H),
                   revolution=rev, has_section=sect, watertight=m.is_watertight,
                   mcb_source=Path(src).name.split("__")[0],
                   mass_casting_fact_SYNTH=round(mass * rng.uniform(1.25, 1.6), 1),
                   labor_hours_fact_SYNTH=round(5 + mass * rng.uniform(0.8, 2.5), 1),
                   fp=",".join(f"{x:.5f}" for x in fp))
    except Exception as e:
        rec["reason"] = f"{type(e).__name__}: {e}"[:200]
        if os.environ.get("GEN_DEBUG"):
            traceback.print_exc()
    return rec

# ---------------------------------------------------------------- пакет
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="макс. моделей на класс (0 = все)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_missing", action="store_true")
    a = ap.parse_args()
    jobs = []
    for cdir in sorted(p for p in Path(a.src).iterdir() if p.is_dir()):
        files = sorted(cdir.glob("*.obj"))
        if a.limit:
            files = files[:a.limit]
        for i, f in enumerate(files):
            seed = int(hashlib.md5(f.name.encode()).hexdigest()[:8], 16) ^ a.seed
            jobs.append((str(f), cdir.name, int(cdir.name[:2]) * 100000 + i + 1, a.out, seed))
    if a.only_missing:
        jobs = [j for j in jobs if not Path(a.out, "png_views", j[1], f"SYN.{j[1][:2]}.{j[2]:06d}.png").exists()]
        print("missing:", len(jobs))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    from tqdm import tqdm
    recs = []
    with Pool(a.workers, maxtasksperchild=50) as pool:
        for r in tqdm(pool.imap_unordered(process, jobs), total=len(jobs), mininterval=30):
            recs.append(r)
    keys = ["designation", "our_class", "status", "reason", "file_pdf", "file_view", "name", "material",
            "mass_kg", "scale", "format", "L", "W", "H", "revolution", "has_section", "watertight",
            "mcb_source", "mass_casting_fact_SYNTH", "labor_hours_fact_SYNTH", "src_obj", "fp"]
    fn = "inventory_regen.csv" if a.only_missing else "inventory_raw.csv"
    with open(Path(a.out, fn), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(recs)
    print("ok:", sum(r["status"] == "ok" for r in recs), "of", len(recs))

if __name__ == "__main__":
    main()
