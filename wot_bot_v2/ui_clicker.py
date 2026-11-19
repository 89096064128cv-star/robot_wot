"""UI state controller for hangar, queue, battle, and post-battle screens."""

import json
import os
import random
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pydirectinput

import config as cfg


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "ui_coords.json"
INDEX_FILE = BASE_DIR / "tank_index.txt"
TEMPLATES_DIR = BASE_DIR / "templates"

HANGAR_THRESHOLD = 0.70
BUTTON_THRESHOLD = 0.70
UI_POLL_SECONDS = 1.0
TANK_SELECT_DELAY = 1.5
TANK_BUSY_RECHECK_SECONDS = 5.0
BATTLE_LOAD_TIMEOUT = 180.0
BATTLE_CONFIRM_DELAY = 20.0
UNKNOWN_RECOVERY_SECONDS = 180.0
RECOVERY_COOLDOWN = 15.0
ZERO_HP_CONFIRM_FRAMES = 2
EXIT_CONFIRM_DELAY = 0.5


def load_coords():
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def _zero_hp_in_region(screen, monitor, coords):
    """Распознаёт нулевой HP по цифре перед косой чертой в индикаторе."""
    reference = coords.get(
        "reference_resolution",
        [cfg.UI_REFERENCE_WIDTH, cfg.UI_REFERENCE_HEIGHT],
    )
    x, y, width, height = coords.get("health_region", [0, 850, 500, 230])
    left = max(0, round(x * monitor["width"] / reference[0]))
    top = max(0, round(y * monitor["height"] / reference[1]))
    right = min(screen.shape[1], round((x + width) * monitor["width"] / reference[0]))
    bottom = min(screen.shape[0], round((y + height) * monitor["height"] / reference[1]))
    if right <= left or bottom <= top:
        return False

    region = screen[top:bottom, left:right]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    bright = (gray > 165) | ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 130))
    binary = bright.astype(np.uint8) * 255
    binary = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return False

    glyphs = []
    for index, contour in enumerate(contours):
        parent = hierarchy[0][index][3]
        child = hierarchy[0][index][2]
        if parent != -1:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 3 or bh < 8:
            continue
        holes = []
        while child != -1:
            holes.append(cv2.boundingRect(contours[child]))
            child = hierarchy[0][child][0]
        glyphs.append((bx, by, bw, bh, holes))

    zeros = []
    for bx, by, bw, bh, holes in glyphs:
        if not holes or len(holes) > 2 or not 0.48 <= bw / bh <= 1.15:
            continue
        hole_centers_y = [
            hole[1] + hole[3] / 2 for hole in holes
        ]
        if len(holes) == 2 and max(hole_centers_y) - min(hole_centers_y) > bh * 0.22:
            continue
        hx = min(hole[0] for hole in holes)
        hy = min(hole[1] for hole in holes)
        hole_right = max(hole[0] + hole[2] for hole in holes)
        hole_bottom = max(hole[1] + hole[3] for hole in holes)
        hw, hh = hole_right - hx, hole_bottom - hy
        hole_center_x = (hx + hw / 2 - bx) / bw
        hole_center_y = (hy + hh / 2 - by) / bh
        if 0.25 <= hole_center_x <= 0.75 and 0.4 <= hole_center_y <= 0.6:
            zeros.append((bx, by, bw, bh))

    for bx, by, bw, bh in zeros:
        for sx, sy, sw, sh, _ in glyphs:
            gap = sx - (bx + bw)
            vertical_overlap = min(by + bh, sy + sh) - max(by, sy)
            if (
                0 <= gap <= bh * 0.55
                and sh >= bh * 0.65
                and sw <= sh * 0.55
                and vertical_overlap >= bh * 0.5
            ):
                previous_digit = any(
                    0 <= bx - (px + pw) <= bh * 0.5
                    and ph >= bh * 0.65
                    and pw <= ph * 0.9
                    and min(by + bh, py + ph) - max(by, py) >= bh * 0.5
                    for px, py, pw, ph, _ in glyphs
                )
                denominator = sorted(
                    (
                        (gx, gy, gw, gh)
                        for gx, gy, gw, gh, _ in glyphs
                        if gx > sx + sw
                        and gh >= bh * 0.65
                        and gh <= bh * 1.3
                        and min(by + bh, gy + gh) - max(by, gy) >= bh * 0.5
                        and gx - (sx + sw) <= bh * 4.5
                    ),
                    key=lambda glyph: glyph[0],
                )
                consecutive_digits = 0
                previous = None
                for digit in denominator:
                    gap = digit[0] - (sx + sw) if previous is None else digit[0] - (previous[0] + previous[2])
                    if gap > bh * (0.65 if previous is None else 0.75):
                        break
                    consecutive_digits += 1
                    previous = digit
                if not previous_digit and consecutive_digits >= 1:
                    return True
    return False


def _scaled_point(point, monitor):
    coords = load_coords()
    reference = coords.get(
        "reference_resolution",
        [cfg.UI_REFERENCE_WIDTH, cfg.UI_REFERENCE_HEIGHT],
    )
    x = round(point[0] * monitor["width"] / reference[0])
    y = round(point[1] * monitor["height"] / reference[1])
    return monitor["left"] + x, monitor["top"] + y


def _match_template(screen, template_name, threshold):
    template_path = TEMPLATES_DIR / template_name
    if not template_path.is_file():
        return None

    template = cv2.imread(str(template_path))
    if template is None:
        raise ValueError(f"Не удалось прочитать шаблон: {template_path}")
    height, width = template.shape[:2]
    if height > screen.shape[0] or width > screen.shape[1]:
        return None

    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    if score < threshold:
        return None
    return location[0] + width // 2, location[1] + height // 2


def _battle_button_position(screen):
    """Возвращает позицию кнопки только при совпадении шаблона и активном цвете."""
    position = _match_template(screen, "btn_battle.png", BUTTON_THRESHOLD)
    if position is None:
        return None

    template_path = TEMPLATES_DIR / "btn_battle.png"
    template = cv2.imread(str(template_path))
    if template is None:
        raise ValueError(f"Не удалось прочитать шаблон: {template_path}")
    height, width = template.shape[:2]
    center_x, center_y = position
    left = center_x - width // 2
    top = center_y - height // 2
    right, bottom = left + width, top + height
    if left < 0 or top < 0 or right > screen.shape[1] or bottom > screen.shape[0]:
        return None

    hsv = cv2.cvtColor(screen[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    active_color = (
        (((hsv[:, :, 0] <= 35) | (hsv[:, :, 0] >= 170))
         & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 90))
    )
    if np.count_nonzero(active_color) < active_color.size * 0.10:
        return None
    return position


def _tank_is_busy(screen, monitor, coords, point):
    """Ищет текстовый маркер занятости в нижней полосе карточки танка."""
    reference = coords.get(
        "reference_resolution",
        [cfg.UI_REFERENCE_WIDTH, cfg.UI_REFERENCE_HEIGHT],
    )
    offset_x, offset_y, width, height = coords.get(
        "tank_status_region",
        [-72, 25, 144, 22],
    )
    left = round((point[0] + offset_x) * monitor["width"] / reference[0])
    top = round((point[1] + offset_y) * monitor["height"] / reference[1])
    right = round((point[0] + offset_x + width) * monitor["width"] / reference[0])
    bottom = round((point[1] + offset_y + height) * monitor["height"] / reference[1])
    left, top = max(0, left), max(0, top)
    right, bottom = min(screen.shape[1], right), min(screen.shape[0], bottom)
    if right <= left or bottom <= top:
        return False

    region = screen[top:bottom, left:right]
    busy_template = TEMPLATES_DIR / "tank_busy.png"
    if busy_template.is_file() and _match_template(region, "tank_busy.png", 0.72):
        return True

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red_text = (
        ((hsv[:, :, 0] <= 5) | (hsv[:, :, 0] >= 175))
        & (hsv[:, :, 1] >= 120)
        & (hsv[:, :, 2] >= 100)
    )
    light_text = (hsv[:, :, 1] <= 80) & (hsv[:, :, 2] >= 210)
    text_mask = (red_text | light_text).astype(np.uint8) * 255
    text_mask = cv2.morphologyEx(
        text_mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), dtype=np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(text_mask)
    glyphs = [
        (x, y, glyph_width, glyph_height, area)
        for x, y, glyph_width, glyph_height, area in stats[1:count]
        if area >= 4 and 2 <= glyph_width <= 18 and 4 <= glyph_height <= 18
    ]
    if len(glyphs) < 3 or len(glyphs) > 12:
        return False

    leftmost = min(glyph[0] for glyph in glyphs)
    rightmost = max(glyph[0] + glyph[2] for glyph in glyphs)
    topmost = min(glyph[1] for glyph in glyphs)
    bottommost = max(glyph[1] + glyph[3] for glyph in glyphs)
    text_area = sum(glyph[4] for glyph in glyphs)
    return (
        20 <= rightmost - leftmost <= 100
        and 6 <= bottommost - topmost <= 18
        and text_area >= 25
    )


def _human_click(x, y):
    current_x, current_y = pydirectinput.position()
    control_x = (current_x + x) / 2 + random.randint(-20, 20)
    control_y = (current_y + y) / 2 + random.randint(-20, 20)
    steps = 12
    for step in range(1, steps + 1):
        t = step / steps
        move_x = int((1 - t) ** 2 * current_x + 2 * (1 - t) * t * control_x + t**2 * x)
        move_y = int((1 - t) ** 2 * current_y + 2 * (1 - t) * t * control_y + t**2 * y)
        pydirectinput.moveTo(move_x, move_y)
        time.sleep(0.01)
    pydirectinput.moveTo(x, y)
    try:
        pydirectinput.mouseDown(button="left")
        time.sleep(0.08)
    finally:
        pydirectinput.mouseUp(button="left")


class UIController:
    """Polls and drives the game UI; combat stays disabled until queue entry."""

    def __init__(self, release_inputs=None):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.coords = load_coords()
        self.battle_active = False
        self.last_state = None
        self.next_poll = 0.0
        self.next_tank_action = 0.0
        self.last_busy_tank_index = None
        self.selected_tank_index = None
        self.busy_tank_indices = set()
        self.all_tanks_busy = False
        self.next_battle_attempt = 0.0
        self.queue_started = None
        self.screen_left_hangar = None
        self.last_recovery = 0.0
        self.hangar_tank_selected = False
        self.release_inputs = release_inputs
        self.zero_hp_frames = 0
        self.death_exit_pending = False
        self.exit_confirmation_pending = False
        self.exit_confirmation_clicked = False
        self.exit_confirmation_after = 0.0

    def _begin_death_exit(self, now):
        self._state_log("destroyed", "Прочность танка равна 0; прекращаю бой и возвращаюсь в ангар.")
        self.battle_active = False
        self.queue_started = None
        self.zero_hp_frames = 0
        self.death_exit_pending = True
        self.last_recovery = now
        try:
            if self.release_inputs:
                self.release_inputs()
        except Exception as error:
            print(f"[ИНТЕРФЕЙС] Не удалось полностью отпустить управление: {error}")
        finally:
            pydirectinput.press("esc")

    def _state_log(self, state, message):
        if self.last_state != state:
            print(f"[ИНТЕРФЕЙС] {message}")
            self.last_state = state

    def _click(
        self,
        button_name,
        screen,
        template_name=None,
        threshold=BUTTON_THRESHOLD,
        position=None,
        require_template=False,
    ):
        if position is None and template_name:
            position = _match_template(screen, template_name, threshold)
        if position is None:
            if require_template:
                print(f"[ИНТЕРФЕЙС] Кнопка «{button_name}» не распознана как активная; не нажимаю.")
                return False
            point = self.coords.get(button_name)
            if point is None:
                print(f"[ИНТЕРФЕЙС] Не найден шаблон и не заданы координаты кнопки «{button_name}».")
                return False
            position = _scaled_point(point, self.monitor)

        print(f"[ИНТЕРФЕЙС] Нажимаю «{button_name}» ({position[0]}, {position[1]}).")
        _human_click(*position)
        return True

    def _click_battle_button(self, screen, position):
        return self._click(
            "in_battle",
            screen,
            position=position,
            require_template=True,
        )

    def _tank_index(self, count):
        try:
            index = int(INDEX_FILE.read_text(encoding="utf-8").strip()) if INDEX_FILE.exists() else 0
        except ValueError:
            index = 0
        return index % count

    def _select_tank(self, screen):
        tanks = self.coords.get("tanks", [])
        if not tanks:
            print("[ИНТЕРФЕЙС] В ui_coords.json не заданы координаты танков.")
            return False

        start_index = self._tank_index(len(tanks))
        busy_indices = getattr(self, "busy_tank_indices", set())
        for offset in range(len(tanks)):
            index = (start_index + offset) % len(tanks)
            if index in busy_indices:
                continue
            if _tank_is_busy(screen, self.monitor, self.coords, tanks[index]):
                busy_indices.add(index)
                self.busy_tank_indices = busy_indices
                self.last_busy_tank_index = index
                continue

            INDEX_FILE.write_text(str((index + 1) % len(tanks)), encoding="utf-8")
            self.selected_tank_index = index
            x, y = _scaled_point(tanks[index], self.monitor)
            print(f"[ИНТЕРФЕЙС] Выбираю свободный танк №{index + 1} из {len(tanks)}.")
            _human_click(x, y)
            return True

        INDEX_FILE.write_text(str(start_index), encoding="utf-8")
        self.all_tanks_busy = True
        print("[ИНТЕРФЕЙС] Все танки заняты; повторно проверю их через несколько секунд.")
        return False

    def _mark_selected_tank_busy(self, now):
        index = getattr(self, "selected_tank_index", None)
        if index is not None:
            self.busy_tank_indices.add(index)
            self.last_busy_tank_index = index
            print(f"[ИНТЕРФЕЙС] Танк №{index + 1} занят («В бою»); перехожу к следующему.")
        self.selected_tank_index = None
        self.hangar_tank_selected = False
        self.next_tank_action = now + TANK_SELECT_DELAY

    def _reset_tank_scan(self):
        self.last_busy_tank_index = None
        self.selected_tank_index = None
        self.busy_tank_indices = set()
        self.all_tanks_busy = False

    def _poll_screen(self):
        image = np.asarray(self.sct.grab(self.monitor))
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    def update(self, force=False):
        """Update UI state if due; returns whether combat control is enabled."""
        now = time.monotonic()
        if not force and now < self.next_poll:
            return self.battle_active
        self.next_poll = now + UI_POLL_SECONDS
        screen = self._poll_screen()

        rating = _match_template(screen, "btn_close_rating.png", BUTTON_THRESHOLD)
        exit_button = _match_template(screen, "btn_exit.png", BUTTON_THRESHOLD)
        leave_button = _match_template(screen, "btn_leave.png", BUTTON_THRESHOLD)
        hangar = _match_template(screen, "hangar_bg.png", HANGAR_THRESHOLD)
        battle_button = _battle_button_position(screen)

        if self.exit_confirmation_pending:
            if now < self.exit_confirmation_after:
                return False
            self._state_log("leave-menu", "Подтверждаю выход и возврат в ангар.")
            if self._click("leave_battle", screen, "btn_leave.png"):
                self.exit_confirmation_clicked = True
            self.exit_confirmation_pending = False
            return False

        if rating:
            self._state_log("rating", "Появилось окно оценки боя; закрываю.")
            self.battle_active = False
            self.queue_started = None
            self._click("close_rating", screen, "btn_close_rating.png")
            return False

        if exit_button:
            self._state_log("exit-menu", "Открыто меню выхода из боя.")
            self.battle_active = False
            self.queue_started = None
            self.screen_left_hangar = None
            self.hangar_tank_selected = False
            self._reset_tank_scan()
            self.next_battle_attempt = now
            if self._click("exit_to_hangar", screen, "btn_exit.png"):
                self.exit_confirmation_pending = True
                self.exit_confirmation_after = now + EXIT_CONFIRM_DELAY
            return False

        if leave_button:
            self._state_log("leave-menu", "Подтверждаю выход и возврат в ангар.")
            self.battle_active = False
            self.queue_started = None
            self.screen_left_hangar = None
            self.hangar_tank_selected = False
            self.next_battle_attempt = now
            self._click("leave_battle", screen, "btn_leave.png")
            self.exit_confirmation_clicked = True
            return False

        if hangar or battle_button:
            returned_from_battle = (
                self.battle_active or self.death_exit_pending
                or self.exit_confirmation_clicked
            )
            if returned_from_battle or self.queue_started is not None:
                self._state_log("hangar-return", "Ангар обнаружен; бой завершён, возвращаюсь к выбору танка.")
            else:
                self._state_log("hangar", "Обнаружен ангар по фону или кнопке «В бой».")
            self.battle_active = False
            self.death_exit_pending = False
            self.exit_confirmation_pending = False
            self.exit_confirmation_clicked = False
            self.zero_hp_frames = 0
            self.screen_left_hangar = None
            if returned_from_battle:
                self.queue_started = None
                self.hangar_tank_selected = False
                self._reset_tank_scan()
                self.next_battle_attempt = now

            if self.queue_started is not None:
                elapsed = now - self.queue_started
                if elapsed < BATTLE_LOAD_TIMEOUT:
                    return False
                self._state_log(
                    "queue-timeout",
                    "За 3 минуты бой не загрузился; нажимаю Esc и сбрасываю ожидание.",
                )
                pydirectinput.press("esc")
                self.queue_started = None
                self.hangar_tank_selected = False
                self.next_battle_attempt = now + 5.0
                self.last_recovery = now
                return False

            if not self.hangar_tank_selected and now >= self.next_tank_action:
                if self.all_tanks_busy:
                    self.busy_tank_indices.clear()
                    self.all_tanks_busy = False
                if self._select_tank(screen):
                    self.hangar_tank_selected = True
                    self.next_tank_action = now + TANK_SELECT_DELAY
                else:
                    self.next_tank_action = now + (
                        TANK_BUSY_RECHECK_SECONDS
                        if self.all_tanks_busy
                        else TANK_SELECT_DELAY
                    )
                return False

            if now < self.next_tank_action:
                return False

            if now >= self.next_battle_attempt:
                if battle_button is not None and self._click_battle_button(screen, battle_button):
                    self._state_log("queue", "Нажата кнопка «В бой». Ожидаю загрузку боя.")
                    self.queue_started = now
                    self.next_battle_attempt = float("inf")
                elif battle_button is None:
                    self._mark_selected_tank_busy(now)
            return False

        if self.battle_active and self.queue_started is None and not self.death_exit_pending:
            if _zero_hp_in_region(screen, self.monitor, self.coords):
                self.zero_hp_frames += 1
                if self.zero_hp_frames >= ZERO_HP_CONFIRM_FRAMES:
                    self._begin_death_exit(now)
                    return False
            else:
                self.zero_hp_frames = 0
        else:
            self.zero_hp_frames = 0

        self.hangar_tank_selected = False
        if self.queue_started is not None:
            if self.screen_left_hangar is None:
                self.screen_left_hangar = now

            elapsed = now - self.queue_started
            away = now - self.screen_left_hangar
            if away >= BATTLE_CONFIRM_DELAY:
                self.battle_active = True
                self._state_log("battle", "Экран ангара сменился после входа в бой; включаю YOLO.")
                return True
            if elapsed >= BATTLE_LOAD_TIMEOUT:
                self._state_log("queue-timeout", "Загрузка боя затянулась; отменяю очередь клавишей Esc.")
                pydirectinput.press("esc")
                self.queue_started = None
                self.screen_left_hangar = None
                self.last_recovery = now
                return False
            self._state_log("loading", "Загрузка/очередь боя; жду, управление стрельбой отключено.")
            return False

        if self.battle_active:
            self._state_log("battle-unknown", "Экран боя не распознан; продолжаю боевой режим.")
            return True

        self._state_log("unknown", "Экран не распознан; ожидаю ангар или экран боя.")
        if not (
            self.death_exit_pending
            or self.exit_confirmation_pending
            or self.exit_confirmation_clicked
        ) and now - self.last_recovery >= UNKNOWN_RECOVERY_SECONDS:
            if now - self.last_recovery >= RECOVERY_COOLDOWN:
                print("[ИНТЕРФЕЙС] Долго не удаётся определить экран; нажимаю Esc для восстановления.")
                pydirectinput.press("esc")
                self.last_recovery = now
        return False


def main():
    controller = UIController()
    print("[ИНТЕРФЕЙС] Автоматизация ангара запущена. Ctrl+C — остановка.")
    try:
        while True:
            controller.update(force=True)
            time.sleep(UI_POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n[ИНТЕРФЕЙС] Остановка автоматизации интерфейса.")


if __name__ == "__main__":
    main()
