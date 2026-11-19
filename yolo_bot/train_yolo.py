import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Обучение модели YOLOv8 на unified_dataset.")
    parser.add_argument("--model-size", choices=("n", "s"), default="n")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model_path = args.checkpoint or (base_dir / f"yolov8{args.model_size}.pt")
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    data_path = base_dir / "data_new.yaml"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Не найден checkpoint {model_path}. Укажите существующий файл через --checkpoint."
        )
    if not data_path.is_file():
        raise FileNotFoundError(f"Не найден YAML датасета: {data_path}")

    print(f"Загрузка модели YOLOv8{args.model_size}: {model_path}")
    model = YOLO(str(model_path))

    print("Начинаем обучение на unified_dataset...")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=f"wot_tanks_unified_{args.model_size}",
        project=str(base_dir / "runs" / "detect"),
    )

    print(
        "Обучение завершено! Модель сохранена в "
        f"runs/detect/wot_tanks_unified_{args.model_size}/weights/best.pt"
    )


if __name__ == "__main__":
    main()