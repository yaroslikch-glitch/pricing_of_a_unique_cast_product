"""Собирает bvk_kaggle_train.ipynb: код train.py и ruclip_vision.py вшивается в ноутбук через %%writefile."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# БВК: поиск аналогов и классификация чертежей — обучение 3 моделей
**Настройки ноутбука (панель справа → Session options):** Accelerator = **GPU T4 x2**, Internet = **On**, Persistence = любой.
Датасет `bvk-drawings` добавить через **+ Add Input** (справа вверху) → Your Datasets.

**Запуск в фоне:** кнопка **Save Version → Save & Run All (Commit)**. Вкладку можно закрыть, через ~1–1.5 ч
результаты появятся во вкладке **Output** этой версии. Скачать `results_light.zip` (метрики и графики).

Модели: **DINOv2 ViT-B/14** (Meta, Apache-2.0) · **ruCLIP ViT-L/14-336** (Sber AI, Apache-2.0, отечественная) · **ResNet50** (baseline).
Для каждой: baseline без дообучения → дообучение ArcFace. Режим `full` — 8 классов; `open` — один класс скрыт (проверка «нового класса»)."""),
code("""import torch, subprocess, sys, os
print(torch.__version__, '| GPU:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available(), 'GPU не включён: Session options -> Accelerator -> GPU T4 x2'
p = torch.cuda.get_device_properties(0)
assert p.major >= 7, f'{p.name}: эта карта не подходит (P100 не поддерживается новым PyTorch). Выбери GPU T4 x2'"""),
code("""!pip install -q -U timm pytorch-metric-learning 2>&1 | grep -v -i warning | tail -2
import timm, pytorch_metric_learning; print('timm', timm.__version__, '| pml', pytorch_metric_learning.__version__)"""),
code("%%writefile ruclip_vision.py\n" + (H / "ruclip_vision.py").read_text(encoding="utf-8")),
code("%%writefile train.py\n" + (H / "train.py").read_text(encoding="utf-8")),
code("""import glob
hits = glob.glob('/kaggle/input/**/split.csv', recursive=True)
assert hits, 'Датасет не подключён: + Add Input -> Your Datasets -> bvk-drawings'
DATA = os.path.dirname(hits[0]); print('DATA =', DATA)
import pandas as pd; s = pd.read_csv(hits[0]); print(pd.crosstab(s.our_class, s.split))"""),
md("""## Обучение
На 2×T4 две очереди идут параллельно (по одной на видеокарту). `EPOCHS=None` — значения по умолчанию из train.py
(15 для ViT, 20 для ResNet50, с ранней остановкой). Для быстрой пробы поставь `EPOCHS = 2`."""),
code("""MODELS_Q = [  # очередь на GPU0, очередь на GPU1
    [('dinov2', 'full'), ('dinov2', 'open'), ('resnet50', 'full'), ('resnet50', 'open')],
    [('ruclip', 'full'), ('ruclip', 'open')],
]
HOLDOUT = '06_petli_kronshteyny'   # класс, скрываемый в режиме open
EPOCHS = None
import time, threading
ngpu = torch.cuda.device_count()
if ngpu < 2: MODELS_Q = [MODELS_Q[0] + MODELS_Q[1]]
os.makedirs('results', exist_ok=True)

def run_queue(gi, q):
    for m, r in q:
        cmd = [sys.executable, 'train.py', '--data', DATA, '--model', m, '--run', r, '--holdout', HOLDOUT,
               '--out', 'results', '--workers', '2'] + (['--epochs', str(EPOCHS)] if EPOCHS else [])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi))
        with open(f'results/console_{m}_{r}.txt', 'w') as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f'[GPU{gi}] {m}/{r}: ' + ('OK' if rc == 0 else f'ОШИБКА (код {rc}) — см. results/console_{m}_{r}.txt'), flush=True)

t0 = time.time()
th = [threading.Thread(target=run_queue, args=(i, q)) for i, q in enumerate(MODELS_Q)]
[t.start() for t in th]
while any(t.is_alive() for t in th):   # раз в 2 минуты — последние строки логов
    time.sleep(120)
    for f in sorted(glob.glob('results/*/log.txt')):
        last = open(f, encoding='utf-8').read().strip().splitlines()[-1:]
        print(f'{time.time()-t0:5.0f}s {f.split("/")[1]:16s} {last[0] if last else ""}'[:160], flush=True)
[t.join() for t in th]
print(f'Всё обучение: {(time.time()-t0)/60:.1f} мин')"""),
md("## Сводка результатов"),
code("""import json, pandas as pd, matplotlib.pyplot as plt
rows = []
for f in sorted(glob.glob('results/*/summary.json')):
    s = json.load(open(f))
    for stage in ('baseline', 'finetuned'):
        if stage not in s: continue
        m = s[stage]
        rows.append({'модель': s['model'], 'режим': s['run'], 'этап': stage,
                     'Recall@1': m['recall_at_1'], 'Recall@5': m['recall_at_5'], 'Accuracy': m['accuracy'],
                     'Macro-F1': m['macro_f1'], 'AUROC нов.класс': m.get('auroc_new_class'),
                     'нов.класс найден': m.get('new_detected'), 'известн. сохранены': m.get('known_kept'),
                     'мин': s['minutes']})
T = pd.DataFrame(rows).round(3)
T.to_csv('results/summary_all.csv', index=False, encoding='utf-8-sig')
display(T)
full = T[T['режим'] == 'full'].pivot(index='модель', columns='этап', values=['Recall@1', 'Macro-F1'])
ax = full.plot.bar(figsize=(9, 4), rot=0, ylim=(0, 1), title='8 классов: до и после дообучения')
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig('results/compare_full.png', dpi=130); plt.show()
op = T[(T['режим'] == 'open')].pivot(index='модель', columns='этап', values='AUROC нов.класс')
if len(op):
    ax = op.plot.bar(figsize=(7, 3.5), rot=0, ylim=(0.4, 1), title=f'Распознавание нового класса (скрыт {HOLDOUT}), AUROC')
    plt.tight_layout(); plt.savefig('results/compare_open.png', dpi=130); plt.show()"""),
code("""from IPython.display import Image as Im, display
for f in sorted(glob.glob('results/*_full/cm_finetuned.png')) + sorted(glob.glob('results/*_full/analogs_finetuned.png'))[:1]:
    print(f); display(Im(f, width=700))"""),
code("""# лёгкий архив: метрики, графики, логи (без весов); веса best.pt и gallery.npz лежат в Output отдельно
!cd results && zip -q -r ../results_light.zip . -x '*.pt' '*.npz' && cd .. && ls -la results_light.zip
!du -sh results/*/best.pt 2>/dev/null"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
for c in nb["cells"]:   # формат ipynb: source — список строк
    c["source"] = c["source"].splitlines(keepends=True)
(H / "bvk_kaggle_train.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")
