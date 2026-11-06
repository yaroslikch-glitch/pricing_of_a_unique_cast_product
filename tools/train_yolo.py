# -*- coding: utf-8 -*-
"""
Скрипт обучения и инференса YOLO для детекции элементов чертежей ЕСКД (ООО "БВК" / ЮУрГУ)
"""
import os
import argparse
from ultralytics import YOLO

def train(data_yaml="dataset_yolo/dataset.yaml", model_name="yolov8n.pt", epochs=30, imgsz=640, device="cpu"):
    print(f"=== ОБУЧЕНИЕ YOLO ДЛЯ ЧЕРТЕЖЕЙ ===")
    print(f"Конфиг данных: {data_yaml}")
    print(f"Базовая модель: {model_name}")
    print(f"Эпох: {epochs}, Устройство: {device}")

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project="runs/detect",
        name="eskd_yolo",
        exist_ok=True,
        save=True,
        plots=True
    )
    print("\n[ГОТОВО] Обучение завершено!")
    print(f"Лучшие веса сохранены в: runs/detect/eskd_yolo/weights/best.pt")
    return results

def predict_and_crop(model_weights, image_path, output_dir="yolo_crops", conf=0.35, device="cpu"):
    """
    Принимает чертеж, детектирует элементы и автоматически вырезает:
      - views_field (для передачи в DINOv2 на поиск аналогов)
      - stamp (для передачи в OCR / Qwen-VL на считывание марки стали, массы)
    """
    print(f"\n-> Анализ чертежа: {image_path}")
    model = YOLO(model_weights)
    results = model.predict(source=image_path, conf=conf, device=device, save=True, project=output_dir, name="predict")

    os.makedirs(output_dir, exist_ok=True)
    for r in results:
        # Сохраняем вырезанные области (crops)
        r.save_crop(save_dir=output_dir, file_name="crop")
        print(f"   Вырезанные элементы сохранены в папку: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение и запуск YOLO на чертежах ЕСКД")
    parser.add_argument("--mode", default="train", choices=["train", "predict"], help="Режим работы")
    parser.add_argument("--data", default="dataset_yolo/dataset.yaml", help="Путь к dataset.yaml")
    parser.add_argument("--weights", default="yolov8n.pt", help="Веса модели (.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Количество эпох")
    parser.add_argument("--device", default="cpu", help="Устройство (cpu или 0 для GPU)")
    parser.add_argument("--source", default=None, help="Изображение для инференса в режиме predict")
    args = parser.parse_args()

    if args.mode == "train":
        train(data_yaml=args.data, model_name=args.weights, epochs=args.epochs, device=args.device)
    else:
        if not args.source:
            print("Укажите путь к чертежу через флаг --source")
        else:
            predict_and_crop(args.weights, args.source, device=args.device)
