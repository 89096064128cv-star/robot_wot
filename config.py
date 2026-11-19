import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
YOLO_MODEL_PATH = os.path.join(
    BASE_DIR, "yolo_bot", "runs", "detect", "wot_tanks_final-2", "weights", "best.pt"
)

# Разрешение 960x540 (16:9)
IMG_W, IMG_H = 960, 540  

# W, A, S, D, Shift, Space, ЛКМ
KEYS = ["w", "a", "s", "d", "shift", "space", "left"]

FPS = 30

# Настройки захвата экрана для основного бота (1920x1080).
SCREEN_W, SCREEN_H = 1920, 1080
UI_REFERENCE_WIDTH, UI_REFERENCE_HEIGHT = SCREEN_W, SCREEN_H
CAPTURE_SIZE = 800
LEFT = (SCREEN_W - CAPTURE_SIZE) // 2
TOP = (SCREEN_H - CAPTURE_SIZE) // 2
MONITOR = {
    "top": TOP,
    "left": LEFT,
    "width": CAPTURE_SIZE,
    "height": CAPTURE_SIZE,
}

# Настройки YOLO.
YOLO_DEVICE = "0"
YOLO_IMGSZ = 640
CONF_THRESHOLD = 0.5
ENEMY_CLASS_ID = 2

# Наведение мыши.
AIM_SMOOTH = 0.35
AIM_DEADZONE = 2
AIM_GAIN = 1.5
AIM_MAX_STEP = 25
FIRE_RADIUS = 40

# Глобальные горячие клавиши и перезарядка.
PAUSE_HOTKEY = "f10"
STOP_HOTKEY = "f12"
RELOAD_TIME = 1.2
TRACK_HOLD_TIME = 1.0