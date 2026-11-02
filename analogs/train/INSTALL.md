# Установка на ПК с RTX 2070 (Windows 10/11)

Время: ~30–40 минут, из них большая часть — скачивание (~3 ГБ PyTorch + ~2.5 ГБ моделей).
Место на диске: ~10 ГБ. Интернет нужен только на шагах 3–5; дальше всё работает офлайн.

## 1. Папка проекта
Создай `C:\bvk\` и распакуй туда этот архив, чтобы получилось `C:\bvk\train\setup_check.py`.
Путь — латиницей и без пробелов (некоторые библиотеки ломаются на `claude folder` и кириллице).

## 2. Драйвер NVIDIA и Python
1. Обнови драйвер видеокарты (GeForce Experience / NVIDIA App или nvidia.com → Drivers).
   Проверка: `Win+R → cmd → nvidia-smi` — в правом верхнем углу «CUDA Version» — запомни это число (например 13.0).
2. Поставь Python 3.12 с python.org. В установщике **поставь галочку «Add python.exe to PATH»**.
   Проверка: `python --version` → `Python 3.12.x`.

## 3. Виртуальное окружение и PyTorch с CUDA
В `cmd`:
```
cd C:\bvk\train
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```
Дальше открой https://pytorch.org/get-started/locally/ и выбери:
**Stable · Windows · Pip · Python · CUDA** — версию CUDA бери **не выше**, чем «CUDA Version» из `nvidia-smi`. Скопируй команду — она вида
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```
и выполни её. RTX 2070 поддерживается всеми CUDA-сборками на сайте.

**Не ставь просто `pip install torch`** — на Windows это CPU-версия без видеокарты.

## 4. Остальные библиотеки
```
pip install -r requirements.txt
```
Пакет `ruclip` ставить **не нужно**: он тянет устаревшие зависимости и не импортируется
с современными библиотеками. Вместо него — `ruclip_vision.py` (тот же визуальный энкодер
ruCLIP, проверен на совпадение выходов с оригиналом).

## 5. Скачивание моделей и проверка
```
python setup_check.py
```
Скрипт сам:
- проверит, что PyTorch видит видеокарту и fp16 работает;
- скачает DINOv2 ViT-B/14, ruCLIP ViT-L/14-336 и ResNet50 в `C:\bvk\train\models\`;
- сделает пробные шаги обучения и подберёт батч, который влезает в 8 ГБ.

В конце — таблица «ИТОГ». **Пришли её мне целиком** — по ней настрою батчи в `train.py`.


## Если что-то пошло не так
| Симптом | Причина и решение |
|---|---|
| `torch.cuda.is_available() == False` | Поставлен CPU-torch: `pip uninstall torch torchvision`, затем шаг 3 заново. Или старый драйвер — шаг 2. |
| `'python' не является командой` | Не поставлена галочка PATH — переустанови Python. |
| Скачивание моделей висит или ошибка соединения | Перед запуском: `set HF_ENDPOINT=https://hf-mirror.com`, затем снова `python setup_check.py`. |
| `OutOfMemoryError` в итоге у всех моделей | Закрой игры и браузер с видео, они занимают видеопамять. |
| Любая другая ошибка | Пришли полный текст вывода. |

## Перенос на другой ПК / Kaggle
Папка `models\` самодостаточна: её можно скопировать на флешку, и `python setup_check.py --offline`
проверит всё без интернета. На Kaggle эти же скрипты работают как есть (включить Internet и GPU T4).
