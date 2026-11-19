"""Ручная запись маршрута по положению танка на миникарте."""

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path


# Область совпадает с миникартой из minimap_navigator.py (1920x1080).
MINIMAP_REGION = {"left": 1578, "top": 743, "width": 325, "height": 325}
LOWER_WHITE = (200, 200, 200)
UPPER_WHITE = (255, 255, 255)
RECORD_INTERVAL = 0.2
MIN_DIST = 3
START_KEY = "f6"
STOP_KEY = "f7"
ROUTES_FILE = Path(__file__).with_name("routes.json")

SPAWN_NAMES = {
    "верхняя": "top_spawn",
    "верхний": "top_spawn",
    "top": "top_spawn",
    "нижняя": "bottom_spawn",
    "нижний": "bottom_spawn",
    "bottom": "bottom_spawn",
}


def find_tank_position(minimap_img):
    """Возвращает центр самого крупного белого маркера или None."""
    import cv2
    import numpy as np

    mask = cv2.inRange(
        minimap_img, np.array(LOWER_WHITE), np.array(UPPER_WHITE)
    )
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 5:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )


def normalize_coords(cx, cy, width, height):
    """Преобразует пиксельные координаты миникарты в диапазон 0..1."""
    if width <= 0 or height <= 0:
        raise ValueError("Размер миникарты должен быть положительным.")
    return round(cx / width, 4), round(cy / height, 4)


def record_point(points, point, width, height, min_dist=MIN_DIST):
    """Добавляет нормализованную точку, если танк достаточно переместился."""
    if point is None:
        return False
    normalized = normalize_coords(point[0], point[1], width, height)
    if points:
        last_point = points[-1]
        dx = (normalized[0] - last_point[0]) * width
        dy = (normalized[1] - last_point[1]) * height
        if math.hypot(dx, dy) < min_dist:
            return False
    points.append(normalized)
    return True


def _validated_points(points):
    if not points:
        raise ValueError("Маршрут должен содержать хотя бы одну точку.")

    validated = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Каждая точка маршрута должна содержать x и y.")
        x, y = (float(value) for value in point)
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Координаты маршрута должны быть в диапазоне 0..1.")
        validated.append([x, y])
    return validated


def save_route(map_name, spawn, points, routes_file=ROUTES_FILE, overwrite=False):
    """Без потери остальных записей добавляет маршрут в существующий routes.json."""
    map_name = map_name.strip()
    if not map_name:
        raise ValueError("Название карты не может быть пустым.")
    if spawn not in ("top_spawn", "bottom_spawn"):
        raise ValueError("Неизвестная сторона респауна.")
    validated = _validated_points(points)
    path = Path(routes_file)

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            routes = json.load(file)
        if not isinstance(routes, dict):
            raise ValueError("Корень routes.json должен быть JSON-объектом.")
    else:
        routes = {}

    map_routes = routes.get(map_name, {})
    if not isinstance(map_routes, dict):
        raise ValueError(f"Запись карты «{map_name}» имеет неверный формат.")
    if spawn in map_routes and not overwrite:
        raise FileExistsError(
            f"Маршрут {map_name}/{spawn} уже существует; требуется подтверждение замены."
        )

    updated_routes = dict(routes)
    updated_map_routes = dict(map_routes)
    updated_map_routes[spawn] = validated
    updated_routes[map_name] = updated_map_routes

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(updated_routes, file, indent=4, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _ask_map_and_spawn():
    map_name = input("Введите название карты (например, berlin): ").strip()
    while not map_name:
        print("Название карты не может быть пустым.")
        map_name = input("Название карты: ").strip()

    while True:
        side = input("Респаун (верхний/top или нижний/bottom): ").strip().lower()
        if side in SPAWN_NAMES:
            return map_name, SPAWN_NAMES[side]
        print("Введите «верхний» или «нижний» (также top/bottom).")


def record_route(
    start_key=START_KEY,
    stop_key=STOP_KEY,
    routes_file=ROUTES_FILE,
    region=MINIMAP_REGION,
):
    """Запускает ручную запись, затем спрашивает карту и респаун в консоли."""
    import cv2
    import keyboard
    import mss
    import numpy as np

    print("=== Ручная запись маршрута ===")
    print(f"Начало: {start_key.upper()} | остановка: {stop_key.upper()}")
    print("Клавиши не меняют основной цикл бота. Ctrl+C отменяет запись.")
    print("Название карты и сторона респауна будут запрошены после остановки.")
    print("Ожидание начала записи…")
    keyboard.wait(start_key)

    points = []
    print("Запись началась. Двигайтесь по маршруту; для остановки нажмите "
          f"{stop_key.upper()}.")
    with mss.mss() as screen:
        while not keyboard.is_pressed(stop_key):
            frame = np.array(screen.grab(region))
            minimap = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            point = find_tank_position(minimap)
            if record_point(
                points, point, region["width"], region["height"]
            ):
                print(f"Точка {len(points)}: {points[-1]}")
            time.sleep(RECORD_INTERVAL)

    if not points:
        print("Маркер танка не найден; маршрут не сохранён.")
        return

    print(f"Запись остановлена: собрано точек — {len(points)}.")
    map_name, spawn = _ask_map_and_spawn()
    try:
        save_route(map_name, spawn, points, routes_file)
    except FileExistsError as error:
        print(error)
        answer = input("Заменить только этот маршрут? (д/н): ").strip().lower()
        if answer not in ("д", "да", "y", "yes"):
            print("Сохранение отменено; существующий маршрут не изменён.")
            return
        save_route(map_name, spawn, points, routes_file, overwrite=True)
    print(f"Маршрут сохранён: {routes_file} → {map_name}/{spawn}.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Записывает реальные позиции танка на миникарте и добавляет "
            "маршрут в wot_bot_v2/routes.json."
        ),
        epilog=(
            "Запуск из корня проекта: python wot_bot_v2\\route_recorder.py. "
            "По умолчанию F6 начинает запись, F7 останавливает."
        ),
    )
    parser.add_argument("--start-key", default=START_KEY, help="клавиша старта (по умолчанию F6)")
    parser.add_argument("--stop-key", default=STOP_KEY, help="клавиша остановки (по умолчанию F7)")
    args = parser.parse_args(argv)
    if args.start_key.lower() == args.stop_key.lower():
        parser.error("клавиши начала и остановки должны различаться")
    record_route(args.start_key, args.stop_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
