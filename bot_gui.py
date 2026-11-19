"""Минимальный графический интерфейс управления ботом."""

import sys
import traceback

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from inference import BotControlState, run


class BotWorker(QThread):
    failed = pyqtSignal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state

    def run(self):
        try:
            run(self.state)
        except Exception as error:
            traceback.print_exc()
            self.failed.emit(str(error))


class ControlWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Управление ботом World of Tanks")
        self.setMinimumWidth(360)

        self.state = None
        self.worker = None
        self.error_message = None
        self.close_when_stopped = False

        self.status = QLabel("Бот остановлен")
        self.start_button = QPushButton("Старт")
        self.pause_button = QPushButton("Пауза")
        self.stop_button = QPushButton("Стоп")

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.stop_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def _start(self):
        if self.worker is not None and self.worker.isRunning():
            return

        self.state = BotControlState()
        self.error_message = None
        self.worker = BotWorker(self.state)
        self.worker.failed.connect(self._show_error)
        self.worker.finished.connect(self._on_finished)
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.status.setText("Запуск бота…")
        self.worker.start()

    def _toggle_pause(self):
        if self.state is None or self.state.stop:
            return
        self.state.paused = not self.state.paused
        if self.state.paused:
            self.pause_button.setText("Продолжить")
            self.status.setText("Пауза — удерживаемые клавиши отпущены")
        else:
            self.pause_button.setText("Пауза")
            self.status.setText("Бот работает")

    def _stop(self):
        if self.state is None or self.state.stop:
            return
        self.state.stop = True
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status.setText("Остановка бота…")

    def _show_error(self, message):
        self.error_message = f"Ошибка: {message}"
        self.status.setText(self.error_message)

    def _on_finished(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("Пауза")
        if self.error_message is None:
            self.status.setText("Бот остановлен")
        else:
            self.status.setText(self.error_message)
        if self.close_when_stopped:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.close_when_stopped = True
            self.state.stop = True
            self.status.setText("Остановка бота перед закрытием…")
            event.ignore()
            return
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ControlWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
