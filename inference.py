"""Боевой цикл YOLO, связанный с ангаром и послебоевой автоматизацией интерфейса."""

import atexit
import ctypes
import errno
import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pydirectinput
from ultralytics import YOLO

import config as cfg
from wot_bot_v2.ui_clicker import UIController, UI_POLL_SECONDS


MODEL_PATH = Path(cfg.YOLO_MODEL_PATH)
CAPTURE_SIZE = cfg.CAPTURE_SIZE
MONITOR = cfg.MONITOR
MOUSEEVENTF_MOVE = 0x0001


class BotControlState:
    """Потокобезопасное состояние запуска, паузы и остановки бота."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()

    @property
    def stop(self):
        return self._stop_event.is_set()

    @stop.setter
    def stop(self, value):
        if value:
            self._stop_event.set()
            self._resume_event.set()
        else:
            self._stop_event.clear()

    @property
    def paused(self):
        return not self._resume_event.is_set()

    @paused.setter
    def paused(self, value):
        if value:
            self._resume_event.clear()
        else:
            self._resume_event.set()

    def wait_for_resume(self, timeout):
        return self._resume_event.wait(timeout)


def _acquire_instance_lock():
    """Не допускает параллельный запуск бота из GUI и скриптов."""
    project_id = hashlib.sha256(
        str(Path(__file__).resolve().parent).encode("utf-8")
    ).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"robot_wot_{project_id}.lock"
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except OSError as error:
        os.close(descriptor)
        lock_errors = (errno.EACCES, errno.EAGAIN, getattr(errno, "EDEADLK", errno.EACCES))
        if error.errno in lock_errors:
            raise RuntimeError("Уже запущен другой экземпляр бота.") from error
        raise


def _release_instance_lock(descriptor):
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def move_mouse(dx, dy):
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)


def run(state=None):
    state = state or BotControlState()
    lock_descriptor = _acquire_instance_lock()
    pressed = set()

    def release_all():
        for key in list(pressed):
            try:
                if key == "left":
                    pydirectinput.mouseUp(button="left")
                else:
                    pydirectinput.keyUp(key)
                pressed.discard(key)
            except Exception as error:
                print(f"[БОТ] Не удалось отпустить «{key}»: {error}")

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Файл YOLO-модели не найден: {MODEL_PATH}")

        print(f"Загрузка модели YOLO: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        print(f"Модель загружена. Классы: {model.names}")

        screen = mss.mss()
        ui = UIController(release_inputs=release_all)
        atexit.register(release_all)
        frame_interval = 1.0 / cfg.FPS
        next_ui_poll = 0.0
        next_no_target_log = 0.0

        print("[БОТ] Запущена связка: ангар → очередь → бой → возврат в ангар.")
        if not hasattr(state, "wait_for_resume"):
            print("[БОТ] Ctrl+C — остановка.")

        while True:
            while state.paused and not state.stop:
                release_all()
                state.wait_for_resume(0.1)
            if state.stop:
                break

            start = time.monotonic()
            if start >= next_ui_poll:
                battle_active = ui.update(force=True)
                next_ui_poll = start + UI_POLL_SECONDS
            else:
                battle_active = ui.battle_active

            if state.stop:
                break
            if state.paused:
                release_all()
                continue

            if not battle_active:
                release_all()
                time.sleep(0.05)
                continue

            image = np.asarray(screen.grab(MONITOR))
            frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            result = model(frame, conf=cfg.CONF_THRESHOLD, imgsz=cfg.YOLO_IMGSZ,
                           device=cfg.YOLO_DEVICE, verbose=False)[0]

            if state.stop:
                break
            if state.paused:
                release_all()
                continue

            center_x = center_y = CAPTURE_SIZE / 2
            best_target = None
            min_distance = float("inf")

            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) != cfg.ENEMY_CLASS_ID:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    target_x = (x1 + x2) / 2
                    target_y = (y1 + y2) / 2
                    distance = ((target_x - center_x) ** 2 + (target_y - center_y) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        best_target = (target_x, target_y)

            if best_target:
                target_x, target_y = best_target
                dx = target_x - center_x
                dy = target_y - center_y
                move_x = int(np.clip(dx * cfg.AIM_GAIN, -cfg.AIM_MAX_STEP, cfg.AIM_MAX_STEP))
                move_y = int(np.clip(dy * cfg.AIM_GAIN, -cfg.AIM_MAX_STEP, cfg.AIM_MAX_STEP))
                if abs(move_x) > cfg.AIM_DEADZONE or abs(move_y) > cfg.AIM_DEADZONE:
                    move_mouse(move_x, move_y)

                if abs(dx) < cfg.FIRE_RADIUS and abs(dy) < cfg.FIRE_RADIUS:
                    if "left" not in pressed:
                        pydirectinput.mouseDown(button="left")
                        pressed.add("left")
                elif "left" in pressed:
                    pydirectinput.mouseUp(button="left")
                    pressed.discard("left")
            elif "left" in pressed:
                pydirectinput.mouseUp(button="left")
                pressed.discard("left")

            if not best_target and start >= next_no_target_log:
                print("[БОТ] Бой активен; врагов в зоне захвата не обнаружено.")
                next_no_target_log = start + 5.0

            elapsed = time.monotonic() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    except KeyboardInterrupt:
        print("\n[БОТ] Остановка по Ctrl+C.")
    finally:
        release_all()
        try:
            _release_instance_lock(lock_descriptor)
        finally:
            print("[БОТ] Все удерживаемые кнопки отпущены.")


if __name__ == "__main__":
    run()
