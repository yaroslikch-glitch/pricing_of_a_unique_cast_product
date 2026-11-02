"""Обучение и оценка 3 моделей для поиска аналогов и классификации чертежей.

Схема (одинакова для всех моделей):
  изображение видов -> backbone -> эмбеддинг (L2-норм.) -> kNN по базе (галерее) ->
  класс голосованием top-k, аналоги = top-5 соседей, «новый класс» = сходство top-1 ниже порога.

Для каждой модели:
  1) baseline: предобученная модель без дообучения (kNN по её признакам);
  2) fine-tune: дообучение последних блоков + проекции 512 с ArcFace-лоссом.

Режимы (--run):
  full — 8 классов; основные метрики (Recall@1/5, accuracy, macro-F1, матрица ошибок) + модель для демо;
  open — один класс (--holdout) полностью скрыт при обучении и в галерее: проверка «нового класса» (AUROC).

Запуск:  python train.py --data <папка со split.csv> --model dinov2 --run full
Модели:  dinov2 | ruclip | resnet50 | siglip2
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HOME", str(HERE / "models" / "hf"))

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIPN = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# name: (вход, нормализация, батч T4/2070, учимых последних блоков (None = вся сеть), lr backbone, epochs)
CFG = {
    "dinov2":   dict(size=448, norm=IMNET, bs=16, last_k=6, lr=2e-5, epochs=15),
    "ruclip":   dict(size=336, norm=CLIPN, bs=16, last_k=4, lr=1e-5, epochs=15),
    "resnet50": dict(size=448, norm=IMNET, bs=32, last_k=None, lr=1e-4, epochs=20),
    "siglip2":  dict(size=384, norm=HALF, bs=16, last_k=4, lr=1e-5, epochs=15),
}


# ============================================================ модели
def build_backbone(name, pretrained=True):
    import timm
    if name == "dinov2":
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, img_size=448)
        return m, m.blocks, m.num_features
    if name == "resnet50":
        m = timm.create_model("resnet50.a1_in1k", pretrained=pretrained, num_classes=0)
        return m, None, m.num_features
    if name == "siglip2":
        m = timm.create_model("vit_base_patch16_siglip_384.v2_webli", pretrained=pretrained, num_classes=0)
        return m, m.blocks, m.num_features
    if name == "ruclip":
        from ruclip_vision import load_ruclip_visual, VisionTransformer, REPO_ID
        if pretrained:
            d = HERE / "models" / "ruclip-vit-large-patch14-336"
            if not (d / "pytorch_model.bin").exists():
                from huggingface_hub import snapshot_download
                snapshot_download(REPO_ID, local_dir=str(d), allow_patterns=["config.json", "pytorch_model.bin"])
            m = load_ruclip_visual(str(d))
        else:  # для отладки без интернета
            m = VisionTransformer(336, 14, 1024, 24, 16, 768)
        return m, m.transformer.resblocks, m.output_dim
    raise ValueError(name)


class Net(nn.Module):
    """backbone -> (опционально) проекция 512 -> L2-нормированный эмбеддинг."""
    def __init__(self, name, pretrained=True, emb=512):
        super().__init__()
        self.backbone, self.blocks, feat = build_backbone(name, pretrained)
        self.proj = nn.Sequential(nn.Linear(feat, emb), nn.BatchNorm1d(emb))
        self.use_proj = False      # baseline: сырые признаки backbone; после fine-tune — проекция

    def forward(self, x):
        f = self.backbone(x).float()
        if self.use_proj:
            f = self.proj(f)
        return F.normalize(f, dim=-1)

    def set_trainable(self, last_k, ckpt=False):
        for p in self.backbone.parameters():
            p.requires_grad = last_k is None
        if last_k is not None:
            for b in list(self.blocks)[-last_k:]:
                for p in b.parameters():
                    p.requires_grad = True
            for n, p in self.backbone.named_parameters():   # финальные нормы/проекции после блоков
                if "blocks" not in n and any(s in n for s in ("ln_post", "proj", "norm", "head", "attn_pool")):
                    p.requires_grad = True
            if ckpt and hasattr(self.backbone, "set_grad_checkpointing"):   # экономия VRAM ценой ~30% скорости
                self.backbone.set_grad_checkpointing(True)


# ============================================================ данные
class Thicken:   # разная толщина линий: имитация другого CAD/сканера
    def __call__(self, im):
        r = random.random()
        if r < 0.25: return im.filter(ImageFilter.MinFilter(3))   # линии толще (фон белый)
        if r < 0.35: return im.filter(ImageFilter.MaxFilter(3))   # тоньше
        return im


class DrawingSet(Dataset):
    def __init__(self, root, df, size, norm, train):
        import torchvision.transforms as T
        self.labels = df.label.values
        self.imgs = [Image.open(root / p).convert("L").copy() for p in df.file_view]   # ~650 МБ в RAM, быстро
        ops = []
        if train:
            ops += [Thicken(),
                    T.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.9, 1.08), fill=255),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.3)]
        ops += [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.Grayscale(3), T.ToTensor(), T.Normalize(*norm)]
        self.tf = T.Compose(ops)

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.tf(self.imgs[i]), self.labels[i]


@torch.no_grad()
def embed(model, loader, dev):
    model.eval(); out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(x.to(dev, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


# ============================================================ метрики
def knn(q, g, k=10):
    s = q @ g.T
    idx = np.argsort(-s, axis=1)[:, :k]
    return idx, np.take_along_axis(s, idx, 1)


def evaluate(q, ql, g, gl, n_cls, k_vote=5):
    from sklearn.metrics import f1_score, confusion_matrix
    idx, sim = knn(q, g, 10)
    nl = gl[idx]
    r1 = float((nl[:, 0] == ql).mean())
    r5 = float((nl[:, :5] == ql[:, None]).any(1).mean())
    votes = np.zeros((len(q), n_cls))
    for j in range(k_vote):
        np.add.at(votes, (np.arange(len(q)), nl[:, j]), np.exp(sim[:, j] / 0.1))   # взвешенное голосование
    pred = votes.argmax(1)
    labels = sorted(set(ql.tolist()))
    return dict(recall_at_1=r1, recall_at_5=r5,
                accuracy=float((pred == ql).mean()),
                macro_f1=float(f1_score(ql, pred, labels=labels, average="macro", zero_division=0)),
                cm=confusion_matrix(ql, pred, labels=list(range(n_cls))).tolist(),
                pred=pred, top1_sim=sim[:, 0], nn_idx=idx, nn_sim=sim)


def open_set(score_known_val, score_unk_val, score_known_test, score_unk_test):
    """score = сходство с ближайшим аналогом. Порог выбираем на val (95% известных проходят), проверяем на test."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(score_known_test)), np.zeros(len(score_unk_test))]
    auroc = roc_auc_score(y, np.r_[score_known_test, score_unk_test])
    thr = float(np.quantile(score_known_val, 0.05))
    return dict(auroc_new_class=float(auroc), threshold=thr,
                known_kept=float((score_known_test >= thr).mean()),
                new_detected=float((score_unk_test < thr).mean()))


# ============================================================ графики
def plot_cm(cm, names, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cm = np.array(cm); n = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if n[i, j] > .5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("предсказано"); ax.set_ylabel("истина"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_analogs(root, qdf, gdf, res, names, path, n=8, seed=0):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed); rows = rng.choice(len(qdf), size=min(n, len(qdf)), replace=False)
    fig, axs = plt.subplots(len(rows), 6, figsize=(12, 2.1 * len(rows)))
    for r, qi in enumerate(rows):
        ax = axs[r, 0]; ax.imshow(Image.open(root / qdf.file_view.iloc[qi]), cmap="gray")
        ax.set_title(f"ЗАПРОС\n{names[qdf.label.iloc[qi]][:18]}", fontsize=7, color="navy")
        for c in range(5):
            gi = res["nn_idx"][qi, c]; ok = gdf.label.iloc[gi] == qdf.label.iloc[qi]
            ax = axs[r, c + 1]; ax.imshow(Image.open(root / gdf.file_view.iloc[gi]), cmap="gray")
            ax.set_title(f"#{c+1} sim={res['nn_sim'][qi, c]:.2f}\n{names[gdf.label.iloc[gi]][:18]}",
                         fontsize=7, color="green" if ok else "red")
        for ax in axs[r]: ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="папка, где лежат split.csv и png_views/")
    ap.add_argument("--model", required=True, choices=list(CFG))
    ap.add_argument("--run", default="full", choices=["full", "open"])
    ap.add_argument("--holdout", default="06_petli_kronshteyny")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int); ap.add_argument("--bs", type=int)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (нужно на 8 ГБ для ruclip)")
    ap.add_argument("--no_finetune", action="store_true", help="только baseline")
    ap.add_argument("--smoke", action="store_true", help="отладка на CPU: без весов, мало данных")
    a = ap.parse_args()

    cfg = dict(CFG[a.model]); cfg["epochs"] = a.epochs or cfg["epochs"]; cfg["bs"] = a.bs or cfg["bs"]
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(a.data)
    out = Path(a.out) / f"{a.model}_{a.run}"; out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a", encoding="utf-8")
    def say(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()

    df = pd.read_csv(root / "split.csv")
    df = df[df.split.isin(["train", "val", "test"])]
    df = df[df.file_view.map(lambda p: (root / p).exists())].reset_index(drop=True)
    names = sorted(df.our_class.unique()); n_cls = len(names)
    df["label"] = df.our_class.map({c: i for i, c in enumerate(names)})
    if a.smoke:
        df = df.groupby(["our_class", "split"]).head(4).reset_index(drop=True); cfg["epochs"] = 1; cfg["bs"] = 4
    hold = names.index(a.holdout) if a.run == "open" else -1
    known = df[df.label != hold]
    tr, va, te = (known[known.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    unk_va = df[(df.label == hold) & (df.split == "val")].reset_index(drop=True)
    unk_te = df[(df.label == hold) & (df.split == "test")].reset_index(drop=True)
    say(f"=== {a.model} / {a.run} | устройство {dev} | train {len(tr)} val {len(va)} test {len(te)}"
        + (f" | скрытый класс {a.holdout}: val {len(unk_va)} test {len(unk_te)}" if hold >= 0 else ""))

    S, N = cfg["size"], cfg["norm"]
    mk = lambda d, t: DrawingSet(root, d, S, N, t)
    dl = lambda ds, sh=False, sampler=None: DataLoader(ds, batch_size=cfg["bs"] * (1 if sampler else 2), shuffle=sh,
                                                       sampler=sampler, num_workers=a.workers, pin_memory=dev == "cuda",
                                                       drop_last=sampler is not None, persistent_workers=a.workers > 0)
    ev = {k: dl(mk(d, False)) for k, d in dict(tr=tr, va=va, te=te, uva=unk_va, ute=unk_te).items() if len(d)}

    model = Net(a.model, pretrained=not a.smoke).to(dev)

    def full_eval(tag):
        E = {k: embed(model, l, dev) for k, l in ev.items()}
        gal = np.r_[E["tr"], E["va"]]; gl = np.r_[tr.label.values, va.label.values]
        gdf = pd.concat([tr, va], ignore_index=True)
        r = evaluate(E["te"], te.label.values, gal, gl, n_cls)
        m = {k: v for k, v in r.items() if k not in ("pred", "top1_sim", "nn_idx", "nn_sim")}
        if hold >= 0:   # порог по val (галерея = train), проверка на test (галерея = train+val)
            rv = evaluate(E["va"], va.label.values, E["tr"], tr.label.values, n_cls)
            su_va = knn(E["uva"], E["tr"], 1)[1][:, 0]; su_te = knn(E["ute"], gal, 1)[1][:, 0]
            m.update(open_set(rv["top1_sim"], su_va, r["top1_sim"], su_te))
        cls_names = [n for i, n in enumerate(names) if i != hold] if hold >= 0 else names
        cm = np.array(m["cm"]); keep = [i for i in range(n_cls) if i != hold]
        plot_cm(cm[np.ix_(keep, keep)] if hold >= 0 else cm, cls_names,
                f"{a.model} {tag}: acc={m['accuracy']:.3f}  R@1={m['recall_at_1']:.3f}", out / f"cm_{tag}.png")
        plot_analogs(root, te, gdf, r, names, out / f"analogs_{tag}.png")
        json.dump(m, open(out / f"metrics_{tag}.json", "w"), indent=1)
        say(f"[{tag}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
        return m, E, gdf

    t0 = time.time()
    base, _, _ = full_eval("baseline")
    results = {"baseline": base}

    if not a.no_finetune:
        model.set_trainable(cfg["last_k"], ckpt=a.ckpt); model.use_proj = True
        from pytorch_metric_learning.losses import ArcFaceLoss
        n_known = n_cls  # метки сохраняем исходные; скрытый класс просто не встречается
        crit = ArcFaceLoss(num_classes=n_known, embedding_size=512, margin=28.6, scale=30).to(dev)
        bb = [p for p in model.backbone.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": bb, "lr": cfg["lr"]},
                                 {"params": model.proj.parameters(), "lr": 1e-3},
                                 {"params": crit.parameters(), "lr": 1e-2}], weight_decay=0.05)
        cnt = tr.label.value_counts(); w = 1.0 / cnt[tr.label].values        # баланс классов
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True)
        trl = dl(mk(tr, True), sampler=sampler)
        steps = cfg["epochs"] * len(trl); warm = len(trl)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
        say(f"fine-tune: учим {sum(p.numel() for p in bb) / 1e6:.1f}M параметров backbone, "
            f"батч {cfg['bs']}, эпох {cfg['epochs']}, шагов/эпоху {len(trl)}")
        best, best_ep, patience = -1, 0, 5
        for ep in range(1, cfg["epochs"] + 1):
            model.train(); te0 = time.time(); tot = 0
            for x, y in trl:
                x, y = x.to(dev, non_blocking=True), y.to(dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    f = model.proj(model.backbone(x).float())
                loss = crit(f.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
                scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
            # валидация: val-запросы против train-галереи
            Etr, Eva = embed(model, ev["tr"], dev), embed(model, ev["va"], dev)
            v = evaluate(Eva, va.label.values, Etr, tr.label.values, n_cls)
            score = v["recall_at_1"] + v["macro_f1"]
            say(f"эпоха {ep:2d}  loss {tot / len(trl):.3f}  val R@1 {v['recall_at_1']:.4f}  "
                f"val F1 {v['macro_f1']:.4f}  {time.time() - te0:.0f} c")
            if score > best:
                best, best_ep = score, ep
                torch.save({"model": a.model, "state": {k: v.half() if v.is_floating_point() else v
                                                        for k, v in model.state_dict().items()},
                            "classes": names, "cfg": cfg}, out / "best.pt")
            elif ep - best_ep >= patience:
                say(f"ранняя остановка: {patience} эпох без улучшения"); break
        ck = torch.load(out / "best.pt", map_location="cpu")
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state"].items()})
        say(f"лучшая эпоха {best_ep}")
        ft, E, gdf = full_eval("finetuned")
        results["finetuned"] = ft; results["best_epoch"] = best_ep
        if a.run == "full":    # база для демо: эмбеддинги всех известных чертежей + путь к файлам
            np.savez_compressed(out / "gallery.npz", emb=np.r_[E["tr"], E["va"], E["te"]].astype(np.float16),
                                files=np.r_[gdf.file_view.values, te.file_view.values].astype(str),
                                labels=np.r_[gdf.label.values, te.label.values], classes=np.array(names, dtype=str))

    results.update(model=a.model, run=a.run, holdout=a.holdout if hold >= 0 else None,
                   minutes=round((time.time() - t0) / 60, 1), cfg={k: v for k, v in cfg.items() if k != "norm"})
    json.dump(results, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    say(f"готово за {results['minutes']} мин -> {out}")


if __name__ == "__main__":
    main()
