"""
Консольная версия бота. Вся логика — в bot_core.py.
Запуск: python bot_cli.py (от имени администратора).
"""

import time

import config as cfg
from bot_core import BotState, run_bot


class StatusPrinter:
    """Печатает статус раз в секунду, чтобы не заспамить консоль."""

    def __init__(self):
        self.last_print = 0.0
        self.frames = 0
        self.shots = 0

    def on_frame(self, frame, targets, best, fired):
        self.frames += 1
        if fired:
            self.shots += 1

        now = time.time()
        if now - self.last_print >= 1.0:
            fps = self.frames / (now - self.last_print) if self.last_print else 0
            status = (
                f"цель №{best['id']} | уверенность: {best['conf']:.2f}"
                if best else "целей нет"
            )
            print(f"Кадров в секунду: {fps:5.1f} | противников: {len(targets)} | "
                  f"{status} | выстрелов: {self.shots}")
            self.last_print = now
            self.frames = 0


if __name__ == "__main__":
    print("=" * 60)
    print("  Бот World of Tanks (консольный режим)")
    print(f"  {cfg.PAUSE_HOTKEY.upper()} — пауза | {cfg.STOP_HOTKEY.upper()} — стоп")
    print("=" * 60)

    state = BotState()
    printer = StatusPrinter()
    run_bot(state, on_frame=printer.on_frame)