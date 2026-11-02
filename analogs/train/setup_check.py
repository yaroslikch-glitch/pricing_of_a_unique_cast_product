"""Шаг 0 перед обучением: проверка окружения + скачивание моделей + тест памяти GPU.

Запуск (из папки проекта, в активированном venv):
    python setup_check.py            # DINOv2, ruCLIP, ResNet50
    python setup_check.py --extra    # + SigLIP 2 (запасная, без дообучения)
    python setup_check.py --offline  # только проверка уже скачанного, без интернета

Все веса кладутся в ./models (рядом со скриптом). Папку models можно скопировать
на флешку / другой ПК — train.py возьмёт веса оттуда без интернета.
"""
import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS / "hf"))          # кэш HuggingFace внутри проекта
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows без прав на симлинки

ap = argparse.ArgumentParser()
ap.add_argument("--extra", action="store_true")
ap.add_argument("--offline", action="store_true")
args = ap.parse_args()
if args.offline:
    os.environ["HF_HUB_OFFLINE"] = "1"

ok = True
def say(tag, msg):
    print(f"[{tag:^5}] {msg}", flush=True)

# ---------------------------------------------------------------- 1. окружение
say("1", f"Python {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    say("FAIL", "Нужен Python 3.10–3.12"); sys.exit(1)
try:
    import torch, torchvision
except ImportError:
    say("FAIL", "PyTorch не установлен. Команду установки возьми на pytorch.org (Windows, Pip, CUDA)."); sys.exit(1)
say("1", f"torch {torch.__version__}, torchvision {torchvision.__version__}, CUDA в сборке: {torch.version.cuda}")
if not torch.cuda.is_available():
    say("FAIL", "torch.cuda.is_available() == False. Причины: поставлен CPU-вариант torch "
                "(в версии нет '+cu'), или старый драйвер NVIDIA. См. инструкцию, шаг 2–3.")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
vram = p.total_memory / 2**30
say("1", f"GPU: {p.name}, {vram:.1f} ГБ, compute capability {p.major}.{p.minor}")
arch = torch.cuda.get_arch_list()
if f"sm_{p.major}{p.minor}" not in arch and f"compute_{p.major}{p.minor}" not in arch:
    say("FAIL", f"Эта сборка torch не поддерживает sm_{p.major}{p.minor}: {arch}. Поставь другую CUDA-сборку."); sys.exit(1)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); t = time.time()
for _ in range(50): a @ a
torch.cuda.synchronize()
say("1", f"fp16 matmul: {50 * 2 * 2048**3 / (time.time() - t) / 1e12:.1f} TFLOPS (RTX 2070 ≈ 20–40)")
if p.major < 8:
    say("info", "bf16 на этой карте нет — обучение пойдёт в fp16 + GradScaler (так и задумано)")

try:
    import timm, huggingface_hub
except ImportError:
    say("FAIL", "pip install timm huggingface_hub"); sys.exit(1)
say("1", f"timm {timm.__version__}, huggingface_hub {huggingface_hub.__version__}")

# ---------------------------------------------------------------- 2. скачивание
sys.path.insert(0, str(ROOT))
from ruclip_vision import load_ruclip_visual, REPO_ID as RUCLIP_REPO

def get_ruclip():
    d = MODELS / "ruclip-vit-large-patch14-336"
    if not (d / "pytorch_model.bin").exists():
        say("2", f"скачиваю {RUCLIP_REPO} (~1.7 ГБ, только config.json + pytorch_model.bin)...")
        huggingface_hub.snapshot_download(RUCLIP_REPO, local_dir=str(d),
                                          allow_patterns=["config.json", "pytorch_model.bin"])
    m = load_ruclip_visual(str(d))
    blocks = m.transformer.resblocks
    return m, blocks, 336

def get_timm(name, size=None):
    say("2", f"загружаю {name} (первый раз — скачивание)...")
    kw = dict(pretrained=True, num_classes=0)
    if size: kw["img_size"] = size
    m = timm.create_model(name, **kw)
    return m

# имя_для_отчёта: (функция загрузки, размер входа, батч, сколько последних блоков учим)
PLAN = {
    "dinov2_vitb14":  (lambda: get_timm("vit_base_patch14_dinov2.lvd142m", 448), 448, 16, 6),
    "ruclip_vitl14":  (get_ruclip, 336, 16, 4),
    "resnet50":       (lambda: get_timm("resnet50.a1_in1k"), 448, 32, None),
}
if args.extra:
    PLAN["siglip2_vitb16"] = (lambda: get_timm("vit_base_patch16_siglip_384.v2_webli"), 384, 16, 4)

# ---------------------------------------------------------------- 3. тест памяти
def train_step_test(name, model, size, bs, last_k):
    model.cuda().train()
    for prm in model.parameters(): prm.requires_grad = last_k is None
    if last_k is not None:
        blocks = model.transformer.resblocks if hasattr(model, "transformer") else model.blocks
        for b in list(blocks)[-last_k:]:
            for prm in b.parameters(): prm.requires_grad = True
        for nm, prm in model.named_parameters():      # финальная норма/проекция
            if any(s in nm for s in ("ln_post", "proj", "norm.", "fc_norm", "head")) and "blocks" not in nm and "resblocks" not in nm:
                prm.requires_grad = True
        if hasattr(model, "set_grad_checkpointing"): model.set_grad_checkpointing(True)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    while bs >= 2:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 3, size, size, device="cuda")
            t = time.time()
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(x).float().pow(2).mean()
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t) / 3
            peak = torch.cuda.max_memory_allocated() / 2**30
            ntr = sum(q.numel() for q in params) / 1e6
            say("OK", f"{name}: вход {size}, батч {bs}, учим {ntr:.0f}M параметров, "
                      f"пик VRAM {peak:.1f} ГБ, {dt:.2f} c/шаг → эпоха 1750 картинок ≈ {1750 / bs * dt / 60:.1f} мин")
            return bs
        except torch.OutOfMemoryError:
            say("warn", f"{name}: батч {bs} не влез в память, пробую {bs // 2}")
            opt.zero_grad(set_to_none=True); bs //= 2
    return 0

result = {}
for name, (loader, size, bs, last_k) in PLAN.items():
    try:
        m = loader()
        with torch.no_grad():
            m.eval().cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                e = m(torch.randn(2, 3, size, size, device="cuda"))
        say("2", f"{name}: загружена, эмбеддинг {tuple(e.shape[1:])}")
        result[name] = train_step_test(name, m, size, bs, last_k)
        del m; torch.cuda.empty_cache()
    except Exception as ex:
        ok = False; result[name] = 0
        say("FAIL", f"{name}: {type(ex).__name__}: {ex}")

print("\n=== ИТОГ ===")
for k, v in result.items():
    print(f"  {k:16s} {'готова, рабочий батч ' + str(v) if v else 'ОШИБКА — пришли этот вывод целиком'}")
size_gb = sum(f.stat().st_size for f in MODELS.rglob("*") if f.is_file()) / 2**30
print(f"  Веса лежат в {MODELS} ({size_gb:.1f} ГБ)")
sys.exit(0 if ok and all(result.values()) else 1)
