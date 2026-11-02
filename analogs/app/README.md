# Демонстрация поиска аналогов (на любом ПК, видеокарта не нужна)

## Что нужно в одной папке, например C:\bvk\demo
- demo.py, stamp_qwen.py — из папки app, train.py и ruclip_vision.py — из папки train (или всё в одной папке);
- best.pt и gallery.npz — с Kaggle (dinov2_full);
- папка с датасетом, где лежат params.csv и png_views\ (распакуй bvk_drawings_kaggle.zip → bvk_drawings).

## Установка (один раз)
```
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm pymupdf scipy pandas pillow
```
По желанию, для чтения штампа со сканов и картинок: Tesseract OCR для Windows (сборка UB Mannheim),
при установке отметить язык Russian и добавить папку Tesseract в PATH. Без него всё работает,
кроме чтения штампа у картинок и PDF без текстового слоя.

## Запуск
```
python demo.py --model best.pt --gallery gallery.npz --data bvk_drawings --out report.html чертёж1.pdf чертёж2.png
```
Откройте report.html в браузере. Одна деталь обрабатывается примерно за 2–4 секунды на процессоре.

Флаги: `--debug_dir dbg` сохраняет вырезанные поля видов (что видит нейросеть);
`--intro файл.html` добавляет вступление в начало отчёта.
