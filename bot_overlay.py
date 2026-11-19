"""
Оверлей-версия бота: YOLO работает в отдельном потоке (QThread),
GUI только рисует рамки. Трекинг — ByteTrack (встроен в ultralytics),
потерянные цели держатся на экране TRACK_HOLD_TIME секунд (серым).
Запуск: python bot_overlay.py (от имени администратора).
"""

import sys
import time

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QFont

import config as cfg
from bot_core import BotState, run_bot


class BotThread(QThread):
    """Рабочий поток: захват + YOLO + управление. GUI не блокируется."""

    data_ready = pyqtSignal(list, float)  # (boxes, fps)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._last_t = time.time()

    def run(self):
        run_bot(self.state, on_frame=self._on_frame)

    def _on_frame(self, frame, targets, best, fired):
        now = time.time()
        fps = 1.0 / max(now - self._last_t, 1e-6)
        self._last_t = now

        best_id = best["id"] if best else None
        boxes = []
        for t in targets:
            boxes.append((
                t["id"],
                int(t["x1"] + cfg.LEFT), int(t["y1"] + cfg.TOP),
                int(t["x2"] + cfg.LEFT), int(t["y2"] + cfg.TOP),
                t["conf"],
                t["id"] == best_id,
            ))
        self.data_ready.emit(boxes, fps)


class Overlay(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state

        # Все флаги одним вызовом (повторный setWindowFlags пересоздаёт окно)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, cfg.SCREEN_W, cfg.SCREEN_H)

        self.fps = 0.0
        # Память рамок: {id: (box_tuple, last_seen_time)}
        self.memory = {}
        # Цели без track id (id == -1) — рисуем только текущий кадр
        self.untracked = []

        # Проверяем флаг остановки, чтобы закрыть приложение по F12
        self.stop_timer = QTimer()
        self.stop_timer.timeout.connect(self._check_stop)
        self.stop_timer.start(200)

    def on_data(self, boxes, fps):
        self.fps = fps
        now = time.time()
        self.untracked = []

        for b in boxes:
            if b[0] >= 0:
                self.memory[b[0]] = (b, now)
            else:
                self.untracked.append(b)

        # Чистим протухшие цели
        for tid in list(self.memory.keys()):
            if now - self.memory[tid][1] > cfg.TRACK_HOLD_TIME:
                del self.memory[tid]

        self.update()

    def _check_stop(self):
        if self.state.stop:
            QApplication.quit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 10))
        now = time.time()

        # Рамка зоны захвата (тонкая, жёлтая)
        painter.setPen(QPen(QColor(255, 255, 0, 80), 1))
        painter.drawRect(cfg.LEFT, cfg.TOP, cfg.CAPTURE_SIZE, cfg.CAPTURE_SIZE)

        # Цели из памяти: свежие — красным, потерянные — серым
        for tid, (box, t_seen) in self.memory.items():
            _, x1, y1, x2, y2, conf, is_best = box
            fresh = (now - t_seen) < 0.15
            if fresh:
                color = QColor(0, 255, 0) if is_best else QColor(255, 0, 0)
            else:
                color = QColor(150, 150, 150, 150)
            painter.setPen(QPen(color, 3 if is_best and fresh else 2))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            label = f"ВРАГ №{tid} | {conf:.2f}" if fresh else f"ЦЕЛЬ ПОТЕРЯНА №{tid}"
            painter.drawText(x1, y1 - 5, label)

        # Цели без track id
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        for (_, x1, y1, x2, y2, conf, is_best) in self.untracked:
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1, y1 - 5, f"ВРАГ | {conf:.2f}")

        # Статусная строка
        if self.state.paused:
            status = f"ПАУЗА ({cfg.PAUSE_HOTKEY.upper()} — продолжить)"
            painter.setPen(QPen(QColor(255, 200, 0), 2))
        else:
            status = (f"БОТ АКТИВЕН | КАДРОВ/С: {self.fps:.0f} | "
                      f"{cfg.PAUSE_HOTKEY.upper()} — пауза | "
                      f"{cfg.STOP_HOTKEY.upper()} — стоп")
            painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(20, 30, status)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    state = BotState()
    overlay = Overlay(state)

    bot_thread = BotThread(state)
    bot_thread.data_ready.connect(overlay.on_data)
    bot_thread.start()

    overlay.show()
    exit_code = app.exec_()

    state.stop = True
    bot_thread.wait(3000)
    sys.exit(exit_code)