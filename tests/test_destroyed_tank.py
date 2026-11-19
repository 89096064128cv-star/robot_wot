import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from wot_bot_v2 import ui_clicker


class DestroyedTankDetectionTests(unittest.TestCase):
    def setUp(self):
        self.monitor = {"width": 1920, "height": 1080, "left": 0, "top": 0}
        self.coords = {
            "reference_resolution": [1920, 1080],
            "health_region": [0, 850, 500, 230],
        }

    def screen_with_health(self, text, color=(255, 255, 255)):
        screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(
            screen, text, (30, 1010), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            color, 2, cv2.LINE_AA,
        )
        return screen

    def test_detects_zero_but_not_nonzero_health(self):
        for text, color, scale, thickness in (
            ("0/900", (255, 255, 255), 1.0, 2),
            ("0/900", (200, 200, 200), 0.65, 1),
            ("0/900", (0, 0, 180), 1.0, 2),
            ("0/1", (255, 255, 255), 0.8, 2),
            ("0/50", (255, 255, 255), 0.8, 2),
            ("0/99", (255, 255, 255), 0.8, 2),
            ("0/100", (255, 255, 255), 0.8, 2),
            ("0/750", (255, 255, 255), 0.8, 2),
            ("0/1000", (255, 255, 255), 1.0, 2),
            ("0/1500", (255, 255, 255), 0.8, 2),
        ):
            with self.subTest(text=text, color=color, scale=scale):
                screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
                cv2.putText(
                    screen, text, (30, 1010), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA,
                )
                self.assertTrue(
                    ui_clicker._zero_hp_in_region(screen, self.monitor, self.coords)
                )
        for health in ("100/900", "10/900", "8/900", "6/900", "9/900"):
            screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(
                screen, health, (30, 1010), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (200, 200, 200), 1, cv2.LINE_AA,
            )
            with self.subTest(health=health):
                self.assertFalse(
                    ui_clicker._zero_hp_in_region(screen, self.monitor, self.coords)
                )

    def test_zero_health_releases_input_once_and_uses_existing_exit_flow(self):
        controller = ui_clicker.UIController.__new__(ui_clicker.UIController)
        controller.monitor = self.monitor
        controller.coords = self.coords
        controller.battle_active = True
        controller.last_state = None
        controller.next_poll = 0
        controller.next_tank_action = 1000
        controller.next_battle_attempt = 0
        controller.queue_started = None
        controller.screen_left_hangar = None
        controller.last_recovery = 0
        controller.hangar_tank_selected = False
        controller.zero_hp_frames = 0
        controller.death_exit_pending = False
        controller.exit_confirmation_pending = False
        controller.exit_confirmation_clicked = False
        controller.exit_confirmation_after = 0
        controller._poll_screen = lambda: np.zeros((1080, 1920, 3), dtype=np.uint8)

        calls = []
        controller.release_inputs = lambda: calls.append("release")
        click_calls = []

        def click(button_name, *_args, **_kwargs):
            click_calls.append(button_name)
            return True

        controller._click = Mock(side_effect=click)
        current_screen = {"template": None}

        def match_template(_screen, template_name, _threshold):
            return (1, 1) if template_name == current_screen["template"] else None

        with (
            patch.object(ui_clicker, "_match_template", side_effect=match_template),
            patch.object(ui_clicker, "_zero_hp_in_region", return_value=True),
            patch.object(ui_clicker.time, "monotonic", side_effect=[10, 11, 250, 251, 252]),
            patch.object(
                ui_clicker.pydirectinput,
                "press",
                side_effect=lambda key: calls.append(key),
            ) as press,
        ):
            self.assertTrue(controller.update(force=True))
            self.assertFalse(controller.update(force=True))
            self.assertEqual(calls, ["release", "esc"])

            current_screen["template"] = "btn_exit.png"
            self.assertFalse(controller.update(force=True))
            current_screen["template"] = None
            self.assertFalse(controller.update(force=True))
            self.assertEqual(click_calls, ["exit_to_hangar", "leave_battle"])

            current_screen["template"] = "hangar_bg.png"
            self.assertFalse(controller.update(force=True))

        self.assertEqual(press.call_count, 1)
        self.assertFalse(controller.battle_active)
        self.assertFalse(controller.death_exit_pending)
        self.assertFalse(controller.exit_confirmation_pending)
        self.assertFalse(controller.exit_confirmation_clicked)
        self.assertEqual(controller._click.call_count, 2)

    def test_confirmation_falls_back_to_leave_button_coordinates_and_releases_mouse(self):
        controller = ui_clicker.UIController.__new__(ui_clicker.UIController)
        controller.monitor = self.monitor
        controller.coords = {
            "reference_resolution": [1920, 1080],
            "leave_battle": [864, 597],
        }
        with (
            patch.object(ui_clicker, "_match_template", return_value=None),
            patch.object(ui_clicker.pydirectinput, "position", return_value=(0, 0)),
            patch.object(ui_clicker.pydirectinput, "moveTo") as move_to,
            patch.object(ui_clicker.pydirectinput, "mouseDown") as mouse_down,
            patch.object(ui_clicker.pydirectinput, "mouseUp") as mouse_up,
            patch.object(ui_clicker.time, "sleep"),
        ):
            self.assertTrue(
                controller._click("leave_battle", np.zeros((1080, 1920, 3), dtype=np.uint8),
                                   "btn_leave.png")
            )

        self.assertEqual(move_to.call_args.args, (864, 597))
        mouse_down.assert_called_once_with(button="left")
        mouse_up.assert_called_once_with(button="left")

    def test_loading_screen_does_not_trigger_death_exit(self):
        controller = ui_clicker.UIController.__new__(ui_clicker.UIController)
        controller.monitor = self.monitor
        controller.coords = self.coords
        controller.battle_active = False
        controller.last_state = None
        controller.next_poll = 0
        controller.next_tank_action = 0
        controller.next_battle_attempt = 0
        controller.queue_started = 10
        controller.screen_left_hangar = 10
        controller.last_recovery = 0
        controller.hangar_tank_selected = False
        controller.zero_hp_frames = 0
        controller.death_exit_pending = False
        controller.exit_confirmation_pending = False
        controller.exit_confirmation_clicked = False
        controller.exit_confirmation_after = 0
        controller._poll_screen = lambda: np.zeros((1080, 1920, 3), dtype=np.uint8)

        with (
            patch.object(ui_clicker, "_match_template", return_value=None),
            patch.object(ui_clicker, "_zero_hp_in_region") as detect_health,
            patch.object(ui_clicker.time, "monotonic", return_value=20),
            patch.object(ui_clicker.pydirectinput, "press") as press,
        ):
            self.assertFalse(controller.update(force=True))

        detect_health.assert_not_called()
        press.assert_not_called()
        self.assertFalse(controller.death_exit_pending)

    def test_hangar_screen_does_not_trigger_death_exit(self):
        controller = ui_clicker.UIController.__new__(ui_clicker.UIController)
        controller.monitor = self.monitor
        controller.coords = self.coords
        controller.battle_active = False
        controller.last_state = None
        controller.next_poll = 0
        controller.next_tank_action = 1000
        controller.next_battle_attempt = 1000
        controller.queue_started = None
        controller.screen_left_hangar = None
        controller.last_recovery = 0
        controller.hangar_tank_selected = True
        controller.zero_hp_frames = 0
        controller.death_exit_pending = False
        controller.exit_confirmation_pending = False
        controller.exit_confirmation_clicked = False
        controller.exit_confirmation_after = 0
        controller._poll_screen = lambda: np.zeros((1080, 1920, 3), dtype=np.uint8)

        def match_template(_screen, template_name, _threshold):
            return (1, 1) if template_name == "hangar_bg.png" else None

        with (
            patch.object(ui_clicker, "_match_template", side_effect=match_template),
            patch.object(ui_clicker, "_zero_hp_in_region") as detect_health,
            patch.object(ui_clicker.time, "monotonic", return_value=20),
            patch.object(ui_clicker.pydirectinput, "press") as press,
        ):
            self.assertFalse(controller.update(force=True))

        detect_health.assert_not_called()
        press.assert_not_called()
        self.assertFalse(controller.death_exit_pending)


if __name__ == "__main__":
    unittest.main()
