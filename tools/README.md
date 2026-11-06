# 🛠️ Служебные утилиты и генераторы данных (Tools)

В данной директории собраны вспомогательные офлайн-скрипты, генераторы синтетических данных, парсеры и утилиты обслуживания проекта.

---

## 📋 Описание утилит

| Скрипт | Назначение | Команда запуска из корня |
| :--- | :--- | :--- |
| **`build_casting_catalog.py`** | Формирование заводского каталога отливок `casting_catalog.json` и предрасчет DINOv2 эмбеддингов `catalog_embeddings.npz`. | `python tools/build_casting_catalog.py` |
| **`generate_yolo_dataset.py`** | Процедурный генератор листов чертежей ЕСКД со штампами, техническими требованиями и разметкой YOLO для обучения детектора. | `python tools/generate_yolo_dataset.py` |
| **`generate_cad_casting_models.py`** | Процедурный 3D-генератор геометрических моделей отливок (фланцы, крыльчатки, втулки, клапаны, валы, кронштейны, плиты). | `python tools/generate_cad_casting_models.py` |
| **`manual_dataset_miner.py`** | Автоматический веб-парсер учебников и методичек ЮУрГУ по литейному производству для расширения датасета. | `python tools/manual_dataset_miner.py` |
| **`train_yolo.py`** | Локальный скрипт запуска обучения/дообучения модели YOLOv8 на процессоре или совместимом GPU. | `python tools/train_yolo.py` |
| **`clean_dataset_junk.py`** | Утилита верификации и санитарной очистки датасета от некорректных или пустых файлов разметки. | `python tools/clean_dataset_junk.py` |
| **`convert_mcb_to_drawings.py`** | Конвертер 3D MCB геометрии в векторные и растровые ортогональные проекции детали. | `python tools/convert_mcb_to_drawings.py` |
| **`audit_dataset.py`** | Инспекция баланса классов и распределения bounding box разметки в датасете `dataset_yolo`. | `python tools/audit_dataset.py` |

---

## 💡 Примечание по запуску
Все скрипты рекомендуется вызывать из корневой директории репозитория:
```powershell
python tools/<имя_скрипта>.py
```