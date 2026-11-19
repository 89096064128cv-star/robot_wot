import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
TARGET_NAMES = ("ally_tank", "dead_tank", "enemy_tank")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMPORT_DIR = BASE_DIR / "imports" / "roboflow_123_bvoul_v2"
STAGING_DIR = BASE_DIR / ".unified_dataset.building"
TARGET_DIR = BASE_DIR / "unified_dataset"
MANIFEST_NAME = "build_manifest.json"


def load_class_names(yaml_path):
    with yaml_path.open("r", encoding="utf-8-sig") as stream:
        dataset_config = yaml.safe_load(stream)
    if not isinstance(dataset_config, dict) or "names" not in dataset_config:
        raise ValueError(f"В YAML не найдены имена классов: {yaml_path}")

    raw_names = dataset_config["names"]
    if isinstance(raw_names, list):
        names = {index: name for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(class_id): name for class_id, name in raw_names.items()}
    else:
        raise ValueError(f"Неподдерживаемый формат names в {yaml_path}")

    if not names or any(not isinstance(name, str) for name in names.values()):
        raise ValueError(f"Некорректные имена классов в {yaml_path}")
    if len(set(names.values())) != len(names):
        raise ValueError(f"Имена классов должны быть уникальны: {yaml_path}")
    return names


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label(label_path, class_names, source_name):
    target_ids = {name: index for index, name in enumerate(TARGET_NAMES)}
    mapped_lines = []
    comparison_rows = []
    content = label_path.read_text(encoding="utf-8-sig")

    for line_number, line in enumerate(content.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Ожидалось 5 столбцов в {label_path}:{line_number}"
            )
        try:
            source_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(
                f"Некорректная строка в {label_path}:{line_number}"
            ) from error
        if source_id not in class_names:
            raise ValueError(
                f"Неизвестный class id {source_id} в {label_path}:{line_number}"
            )

        class_name = class_names[source_id]
        if class_name not in target_ids:
            raise ValueError(
                f"Неизвестный класс {class_name!r} ({source_name}, "
                f"{label_path}:{line_number})"
            )
        x_center, y_center, width, height = coordinates
        if (
            any(
                not math.isfinite(value) or value < 0 or value > 1
                for value in coordinates
            )
            or width <= 0
            or height <= 0
        ):
            raise ValueError(
                f"Координаты вне диапазона в {label_path}:{line_number}"
            )

        target_id = target_ids[class_name]
        mapped_lines.append(
            " ".join([str(target_id)] + [part for part in parts[1:]])
        )
        comparison_rows.append((target_id, *coordinates))

    return "\n".join(mapped_lines) + ("\n" if mapped_lines else ""), tuple(
        sorted(comparison_rows)
    )


def get_sources():
    sources = [
        {
            "name": "dataset_all",
            "root": BASE_DIR / "dataset_all",
            "yaml": BASE_DIR / "data_all.yaml",
        }
    ]

    import_yaml = IMPORT_DIR / "data.yaml"
    import_images = any(
        image_dir.is_dir()
        and any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in image_dir.iterdir()
        )
        for split in ("train", "valid", "test")
        for image_dir in (IMPORT_DIR / split / "images",)
    )
    if import_yaml.is_file():
        sources.append(
            {"name": "roboflow_123_bvoul_v2", "root": IMPORT_DIR, "yaml": import_yaml}
        )
    elif import_images:
        raise FileNotFoundError(
            f"В каталоге с изображениями Roboflow-импорта отсутствует {import_yaml}"
        )
    return sources


def collect_records():
    records_by_hash = defaultdict(list)
    source_summaries = []

    for source in get_sources():
        root = source["root"]
        if not root.is_dir():
            raise FileNotFoundError(f"Не найден локальный источник: {root}")
        source_names = load_class_names(source["yaml"])
        if not set(source_names.values()).issubset(TARGET_NAMES):
            raise ValueError(f"Неизвестное соответствие классов в {source['yaml']}")

        source_image_count = 0
        for split_name in ("train", "valid", "test"):
            image_dir = root / split_name / "images"
            if not image_dir.is_dir():
                continue
            label_dir = root / split_name / "labels"
            if not label_dir.is_dir():
                raise FileNotFoundError(f"Не найдена папка разметок: {label_dir}")

            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(f"Для изображения нет разметки: {image_path}")
                label_text, label_key = parse_label(
                    label_path, source_names, source["name"]
                )
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as error:
                    raise ValueError(f"Не удалось прочитать изображение: {image_path}") from error
                digest = sha256_file(image_path)
                records_by_hash[digest].append(
                    {
                        "source": source["name"],
                        "split": split_name,
                        "image": image_path,
                        "label": label_path,
                        "label_text": label_text,
                        "label_key": label_key,
                    }
                )
                source_image_count += 1

        source_summaries.append(
            {
                "name": source["name"],
                "root": str(root.relative_to(BASE_DIR)),
                "yaml": str(source["yaml"].relative_to(BASE_DIR)),
                "class_names": {
                    str(class_id): name for class_id, name in sorted(source_names.items())
                },
                "images": source_image_count,
            }
        )

    return records_by_hash, source_summaries


def choose_split(digest):
    bucket = int(hashlib.sha256(("robot_wot:" + digest).encode("ascii")).hexdigest()[:8], 16)
    return "train" if bucket < int(0.8 * (2**32)) else "valid"


def build_dataset(replace_generated=False):
    if STAGING_DIR.exists():
        raise FileExistsError(
            f"Существует временный каталог сборки {STAGING_DIR}; проверьте его вручную."
        )
    if TARGET_DIR.exists():
        marker = TARGET_DIR / MANIFEST_NAME
        if not replace_generated or not marker.is_file():
            raise FileExistsError(
                f"Целевой каталог уже существует: {TARGET_DIR}. "
                "Сохраните его или используйте --replace-generated только для "
                "каталога, созданного этим скриптом."
            )

    records_by_hash, source_summaries = collect_records()
    conflict_groups = {
        digest: records
        for digest, records in records_by_hash.items()
        if len({record["label_key"] for record in records}) > 1
    }

    usable_groups = {
        digest: records
        for digest, records in records_by_hash.items()
        if digest not in conflict_groups
    }
    if not usable_groups:
        raise ValueError("После проверки разметок не осталось изображений для сборки.")

    split_counts = Counter()
    class_counts = Counter()
    empty_labels = 0
    STAGING_DIR.mkdir()
    try:
        for digest, records in sorted(usable_groups.items()):
            record = sorted(
                records,
                key=lambda item: (
                    item["source"],
                    item["split"],
                    item["image"].name.lower(),
                    str(item["image"]),
                ),
            )[0]
            split_name = choose_split(digest)
            suffix = record["image"].suffix.lower()
            output_stem = digest
            image_dir = STAGING_DIR / split_name / "images"
            label_dir = STAGING_DIR / split_name / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["image"], image_dir / f"{output_stem}{suffix}")
            (label_dir / f"{output_stem}.txt").write_text(
                record["label_text"], encoding="utf-8"
            )

            split_counts[split_name] += 1
            if record["label_key"]:
                class_counts.update(row[0] for row in record["label_key"])
            else:
                empty_labels += 1

        conflict_report = STAGING_DIR / "duplicate_conflicts.csv"
        with conflict_report.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["sha256", "source", "split", "image_path", "label_path", "mapped_labels"]
            )
            for digest, records in sorted(conflict_groups.items()):
                for record in sorted(records, key=lambda item: str(item["image"])):
                    writer.writerow(
                        [
                            digest,
                            record["source"],
                            record["split"],
                            str(record["image"].relative_to(BASE_DIR)),
                            str(record["label"].relative_to(BASE_DIR)),
                            record["label_text"].replace("\n", " | ").strip(),
                        ]
                    )

        manifest = {
            "target_classes": list(TARGET_NAMES),
            "split_strategy": "SHA-256 content hash, deterministic 80/20 train/valid",
            "sources": source_summaries,
            "unique_images_written": sum(split_counts.values()),
            "duplicate_groups": sum(1 for records in records_by_hash.values() if len(records) > 1),
            "conflicting_duplicate_groups_excluded": len(conflict_groups),
            "split_counts": {name: split_counts[name] for name in ("train", "valid")},
            "class_object_counts": {
                TARGET_NAMES[class_id]: class_counts[class_id]
                for class_id in range(len(TARGET_NAMES))
            },
            "empty_label_images": empty_labels,
        }
        (STAGING_DIR / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if TARGET_DIR.exists():
            backup = BASE_DIR / "unified_dataset.previous"
            if backup.exists():
                raise FileExistsError(
                    f"Нельзя сохранить старый датасет: уже существует {backup}"
                )
            TARGET_DIR.rename(backup)
        STAGING_DIR.rename(TARGET_DIR)
    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)
        raise

    print(f"Создано изображений: {manifest['unique_images_written']}")
    print(f"Разбиение train/valid: {manifest['split_counts']}")
    print(f"Конфликтных групп исключено: {manifest['conflicting_duplicate_groups_excluded']}")
    if not (IMPORT_DIR / "data.yaml").is_file():
        print(
            "Roboflow-экспорт локально не найден; импорт можно положить в "
            f"{IMPORT_DIR.relative_to(BASE_DIR)} и повторить сборку."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Безопасная сборка unified_dataset из локальных YOLO-источников."
    )
    parser.add_argument(
        "--replace-generated",
        action="store_true",
        help="Сохранить прежний созданный этим скриптом датасет как unified_dataset.previous.",
    )
    arguments = parser.parse_args()
    build_dataset(replace_generated=arguments.replace_generated)


if __name__ == "__main__":
    main()
