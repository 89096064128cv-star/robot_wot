import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from wot_bot_v2 import ui_clicker


class HangarTankSelectionTests(unittest.TestCase):
    def setUp(self):
        self.monitor = {"width": 1920, "height": 1080, "left": 0, "top": 0}
        self.tanks = [
            [155, 865],
            [385, 865],
            [615, 865],
            [155, 985],
            [385, 985],
            [615, 985],
        ]
        self.coords = {
            "reference_resolution": [1920, 1080],
            "tank_status_region": [-72, 25, 144, 22],
            "tanks": self.tanks,
        }
        self.controller = ui_clicker.UIController.__new__(ui_clicker.UIController)
        self.controller.monitor = self.monitor
        self.controller.coords = self.coords
        self.controller.last_state = None
        self.controller.last_busy_tank_index = None
        self.controller.selected_tank_index = None
        self.controller.busy_tank_indices = set()
        self.controller.all_tanks_busy = False
        self.screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_file = Path(self.temp_dir.name) / "tank_index.txt"
        self.index_file.write_text("0", encoding="utf-8")
        self.index_file_patch = patch.object(ui_clicker, "INDEX_FILE", self.index_file)
        self.index_file_patch.start()
        self.addCleanup(self.index_file_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def test_skips_busy_first_tank_and_selects_free_second(self):
        clicked = []
        with (
            patch.object(ui_clicker, "_tank_is_busy", side_effect=[True, False]),
            patch.object(ui_clicker, "_scaled_point", side_effect=lambda point, _monitor: tuple(point)),
            patch.object(ui_clicker, "_human_click", side_effect=lambda x, y: clicked.append((x, y))),
        ):
            self.assertTrue(self.controller._select_tank(self.screen))

        self.assertEqual(clicked, [tuple(self.tanks[1])])
        self.assertEqual(self.controller.last_busy_tank_index, 0)
        self.assertEqual(self.controller.selected_tank_index, 1)
        self.assertEqual(self.index_file.read_text(encoding="utf-8"), "2")

    def test_all_tanks_busy_waits_without_clicking(self):
        with (
            patch.object(ui_clicker, "_tank_is_busy", return_value=True),
            patch.object(ui_clicker, "_human_click") as click,
            patch("builtins.print") as print_message,
        ):
            self.assertFalse(self.controller._select_tank(self.screen))

        click.assert_not_called()
        self.assertEqual(self.controller.busy_tank_indices, set(range(6)))
        self.assertEqual(self.controller.last_busy_tank_index, 5)
        self.assertTrue(self.controller.all_tanks_busy)
        self.assertEqual(self.index_file.read_text(encoding="utf-8"), "0")
        self.assertTrue(any("Все танки заняты" in str(call) for call in print_message.call_args_list))

    def test_detects_busy_marker_in_tank_status_region(self):
        for offset in range(5):
            x = 115 + offset * 8
            self.screen[896:906, x:x + 4] = (0, 0, 255)

        self.assertTrue(
            ui_clicker._tank_is_busy(
                self.screen,
                self.monitor,
                self.coords,
                self.tanks[0],
            )
        )
        self.assertFalse(
            ui_clicker._tank_is_busy(
                np.zeros_like(self.screen),
                self.monitor,
                self.coords,
                self.tanks[0],
            )
        )

    def test_normal_hangar_cards_are_not_marked_busy(self):
        screen = ui_clicker.cv2.imread(
            str(ui_clicker.TEMPLATES_DIR / "hangar_bg.png")
        )

        self.assertEqual(
            [
                ui_clicker._tank_is_busy(
                    screen,
                    self.monitor,
                    self.coords,
                    point,
                )
                for point in self.tanks
            ],
            [False] * len(self.tanks),
        )

    def test_gray_battle_button_is_not_clicked(self):
        template = np.zeros((10, 20, 3), dtype=np.uint8)
        with (
            patch.object(ui_clicker, "_match_template", return_value=(100, 50)),
            patch.object(ui_clicker.cv2, "imread", return_value=template),
            patch.object(ui_clicker, "_human_click") as click,
        ):
            position = ui_clicker._battle_button_position(
                np.full((100, 200, 3), 120, dtype=np.uint8)
            )
            self.assertIsNone(position)
            self.assertFalse(self.controller._click_battle_button(self.screen, position))

        click.assert_not_called()

    def test_active_battle_button_is_clicked(self):
        template = np.zeros((10, 20, 3), dtype=np.uint8)
        screen = np.zeros((100, 200, 3), dtype=np.uint8)
        screen[45:55, 90:110] = (0, 100, 255)
        with (
            patch.object(ui_clicker, "_match_template", return_value=(100, 50)),
            patch.object(ui_clicker.cv2, "imread", return_value=template),
            patch.object(ui_clicker, "_human_click") as click,
        ):
            position = ui_clicker._battle_button_position(screen)
            self.assertEqual(position, (100, 50))
            self.assertTrue(self.controller._click_battle_button(screen, position))

        click.assert_called_once_with(100, 50)


if __name__ == "__main__":
    unittest.main()
