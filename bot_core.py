"""
Общее ядро бота: захват экрана, YOLO-детекция с трекингом (ByteTrack),
наведение со сглаживанием, стрельба с перезарядкой, аварийный стоп.
Используется и консольной версией (bot_cli.py), и оверлеем (bot_overlay.py).
"""

import time
import ctypes
import atexit

import cv2
import numpy as np
import mss
import keyboard
import pydirectinput
from ultralytics import YOLO

import config as cfg

pydirectinput.PAUSE = 0  # убираем встроенную задержку pydirectinput


# ================== МЫШЬ ЧЕРЕЗ SendInput (надёжнее mouse_event) ==================

PUL = ctypes.POINTER(ctypes.c_ulong)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    _anonymous_ = ("i",)
    _fields_ = [("type", ctypes.c_ulong), ("i", _I)]


MOUSEEVENTF_MOVE = 0x0001


def move_mouse(dx, dy):
    """Относительное перемещение мыши через SendInput."""
    dx, dy = int(dx), int(dy)
    if dx == 0 and dy == 0:
        return
    inp = INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, None)
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


# ================== СОСТОЯНИЕ И ГОРЯЧИЕ КЛАВИШИ ==================

class BotState:
    """Стоп/пауза через глобальные горячие клавиши (работают поверх игры)."""

    def __init__(self):
        self.stop = False
        self.paused = False
        keyboard.add_hotkey(cfg.STOP_HOTKEY, self._on_stop)
        keyboard.add_hotkey(cfg.PAUSE_HOTKEY, self._on_pause)

    def _on_stop(self):
        self.stop = True
        print(f"\n[{cfg.STOP_HOTKEY.upper()}] Остановка бота...")

    def _on_pause(self):
        self.paused = not self.paused
        print(f"\n[{cfg.PAUSE_HOTKEY.upper()}] "
              + ("ПАУЗА" if self.paused else "ПРОДОЛЖЕНИЕ"))


# ================== УПРАВЛЕНИЕ (клавиши + стрельба) ==================

class Controller:
    def __init__(self):
        self.pressed = set()
        self.last_shot = 0.0
        atexit.register(self.release_all)

    def press(self, key):
        if key not in self.pressed:
            try:
                if key == "left":
                    pydirectinput.mouseDown(button="left")
                else:
                    pydirectinput.keyDown(key)
                self.pressed.add(key)
            except Exception:
                pass

    def release(self, key):
        if key in self.pressed:
            try:
                if key == "left":
                    pydirectinput.mouseUp(button="left")
                else:
                    pydirectinput.keyUp(key)
            except Exception:
                pass
            self.pressed.discard(key)

    def fire(self):
        """Одиночный выстрел с учётом перезарядки. Возвращает True, если выстрелил."""
        now = time.time()
        if now - self.last_shot >= cfg.RELOAD_TIME:
            try:
                pydirectinput.click(button="left")
            except Exception:
                return False
            self.last_shot = now
            return True
        return False

    def release_all(self):
        for k in list(self.pressed):
            self.release(k)


# ================== ДЕТЕКТОР (YOLO + ByteTrack) ==================

class Detector:
    def __init__(self):
        print("Загрузка модели YOLO...")
        self.model = YOLO(cfg.YOLO_MODEL_PATH)
        print("Модель загружена.")

    def detect(self, frame):
        """Возвращает список врагов: [{id, x1, y1, x2, y2, cx, cy, conf}, ...]"""
        results = self.model.track(
            frame,
            conf=cfg.CONF_THRESHOLD,
            imgsz=cfg.YOLO_IMGSZ,
            device=cfg.YOLO_DEVICE,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        targets = []
        if results.boxes is None:
            return targets

        for box in results.boxes:
            if int(box.cls[0]) != cfg.ENEMY_CLASS_ID:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tid = int(box.id[0]) if box.id is not None else -1
            targets.append({
                "id": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "conf": float(box.conf[0]),
            })
        return targets


# ================== НАВЕДЕНИЕ СО СГЛАЖИВАНИЕМ ==================

class Aimer:
    def __init__(self):
        self.sx = 0.0
        self.sy = 0.0

    def reset(self):
        self.sx = 0.0
        self.sy = 0.0

    def update(self, dx, dy):
        """EMA-сглаживание + мёртвая зона + ограничение шага."""
        a = cfg.AIM_SMOOTH
        self.sx = a * dx + (1 - a) * self.sx
        self.sy = a * dy + (1 - a) * self.sy

        mx = 0 if abs(self.sx) < cfg.AIM_DEADZONE else int(
            np.clip(self.sx * cfg.AIM_GAIN, -cfg.AIM_MAX_STEP, cfg.AIM_MAX_STEP))
        my = 0 if abs(self.sy) < cfg.AIM_DEADZONE else int(
            np.clip(self.sy * cfg.AIM_GAIN, -cfg.AIM_MAX_STEP, cfg.AIM_MAX_STEP))
        return mx, my


# ================== ВЫБОР ЦЕЛИ ==================

def select_target(targets, center):
    """Ближайший к центру экрана враг."""
    best, min_dist = None, float("inf")
    for t in targets:
        dist = ((t["cx"] - center) ** 2 + (t["cy"] - center) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best = t
    return best


# ================== ГЛАВНЫЙ ЦИКЛ ==================

def run_bot(state, on_frame=None):
    """
    Главный цикл бота.
    state    — BotState (стоп/пауза).
    on_frame — необязательный колбэк on_frame(frame, targets, best, fired),
               вызывается каждый кадр (используется оверлеем).
    """
    sct = mss.mss()  # mss не потокобезопасен — создаём в рабочем потоке
    detector = Detector()
    controller = Controller()
    aimer = Aimer()

    interval = 1.0 / cfg.FPS
    center = cfg.CAPTURE_SIZE / 2

    print(f"Бот запущен. {cfg.PAUSE_HOTKEY.upper()} — пауза, "
          f"{cfg.STOP_HOTKEY.upper()} — остановка.")

    try:
        while not state.stop:
            t0 = time.time()

            if state.paused:
                controller.release_all()
                aimer.reset()
                if on_frame:
                    on_frame(None, [], None, False)
                time.sleep(0.1)
                continue

            # 1. Захват
            img = np.array(sct.grab(cfg.MONITOR))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 2. Детекция
            targets = detector.detect(frame)
            best = select_target(targets, center)

            # 3. Наведение и стрельба
            fired = False
            if best:
                dx = best["cx"] - center
                dy = best["cy"] - center

                mx, my = aimer.update(dx, dy)
                move_mouse(mx, my)

                if abs(dx) < cfg.FIRE_RADIUS and abs(dy) < cfg.FIRE_RADIUS:
                    fired = controller.fire()
            else:
                aimer.reset()

            # 4. Колбэк для оверлея/логов
            if on_frame:
                on_frame(frame, targets, best, fired)

            # 5. Ограничение FPS
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)
    finally:
        controller.release_all()
        print("Все клавиши отпущены. Бот остановлен.")