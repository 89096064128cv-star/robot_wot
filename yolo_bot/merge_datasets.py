import hashlib
import shutil
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
FINAL_DIR = BASE_DIR / "dataset_all"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def valid_label(label_path, allow_empty=False):
    if not label_path.exists():
        return False
    lines = label_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return allow_empty
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            return False
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            return False
        if class_id not in (0, 1, 2):
            return False
        if any(value < 0 or value > 1 for value in values):
            return False
        if values[2] <= 0 or values[3] <= 0:
            return False
    return True


def split_for_image(image_path):
    digest = hashlib.sha256(str(image_path).encode("utf-8")).digest()
    bucket = digest[0]
    if bucket < 204:
        return "train"
    if bucket < 230:
        return "valid"
    return "test"


def copy_image(image_path, label_path, output_name, custom=False):
    if not cv2.haveImageReader(str(image_path)):
        return False
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    if not valid_label(label_path, allow_empty=custom):
        return False

    split = split_for_image(output_name)
    image_dir = FINAL_DIR / split / "images"
    label_dir = FINAL_DIR / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    destination_image = image_dir / output_name
    shutil.copy2(image_path, destination_image)

    if custom:
        # Custom label scripts identify enemy-like detections as class 0.
        # In the consolidated 3-class dataset that class is enemy_tank (2).
        converted = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                converted.append("2 " + " ".join(parts[1:]))
        label_text = "\n".join(converted) + ("\n" if converted else "")
    else:
        label_text = label_path.read_text(encoding="utf-8")

    (label_dir / output_name.with_suffix(".txt").name).write_text(
        label_text, encoding="utf-8"
    )
    return True


def main():
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)

    sources = []
    custom_images = BASE_DIR / "dataset" / "images"
    custom_labels = BASE_DIR / "dataset" / "labels"
    for image_path in sorted(custom_images.glob("*")):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            sources.append((
                image_path,
                custom_labels / image_path.with_suffix(".txt").name,
                f"custom_{image_path.name}",
                True,
            ))

    for split_name in ("train", "valid", "test"):
        image_dir = BASE_DIR / split_name / "images"
        label_dir = BASE_DIR / split_name / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                sources.append((
                    image_path,
                    label_dir / image_path.with_suffix(".txt").name,
                    f"{split_name}_{image_path.name}",
                    False,
                ))

    imported = 0
    skipped = 0
    for image_path, label_path, output_name, custom in sources:
        if copy_image(image_path, label_path, Path(output_name), custom):
            imported += 1
        else:
            skipped += 1

    counts = {}
    for split_name in ("train", "valid", "test"):
        split_dir = FINAL_DIR / split_name
        images = len(list((split_dir / "images").glob("*"))) if split_dir.exists() else 0
        counts[split_name] = images

    print(f"Объединено изображений: {imported}")
    print(f"Пропущено (нет меток/повреждено): {skipped}")
    print(f"Разделение: {counts}")


if __name__ == "__main__":
    main()