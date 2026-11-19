import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import inference


class BotControlStateTests(unittest.TestCase):
    def test_pause_resume_and_stop_wake_waiting_worker(self):
        state = inference.BotControlState()
        self.assertFalse(state.paused)
        self.assertTrue(state.wait_for_resume(0))

        state.paused = True
        self.assertTrue(state.paused)
        self.assertFalse(state.wait_for_resume(0))

        state.paused = False
        self.assertFalse(state.paused)
        self.assertTrue(state.wait_for_resume(0))

        state.paused = True
        state.stop = True
        self.assertTrue(state.stop)
        self.assertTrue(state.wait_for_resume(0))


class InferenceLifecycleTests(unittest.TestCase):
    def test_pause_releases_held_mouse_and_prevents_next_cycle(self):
        state = inference.BotControlState()
        box = SimpleNamespace(
            cls=[inference.cfg.ENEMY_CLASS_ID],
            xyxy=np.array([[400, 400, 400, 400]]),
            id=None,
            conf=[0.9],
        )
        model = Mock(return_value=[SimpleNamespace(boxes=[box])])
        screen = Mock()
        screen.grab.return_value = object()
        ui = Mock()
        ui.battle_active = True
        ui.update.return_value = True
        model_path = Mock()
        model_path.is_file.return_value = True

        def mouse_down(**_kwargs):
            state.paused = True

        def mouse_up(**_kwargs):
            state.stop = True

        with (
            patch.object(inference, "_acquire_instance_lock", return_value=1),
            patch.object(inference, "_release_instance_lock"),
            patch.object(inference, "MODEL_PATH", model_path),
            patch.object(inference, "YOLO", return_value=model),
            patch.object(inference.mss, "mss", return_value=screen),
            patch.object(inference, "UIController", return_value=ui),
            patch.object(inference.cv2, "cvtColor", return_value=object()),
            patch.object(inference.atexit, "register"),
            patch.object(inference.time, "sleep"),
            patch.object(inference.pydirectinput, "mouseDown", side_effect=mouse_down) as down,
            patch.object(inference.pydirectinput, "mouseUp", side_effect=mouse_up) as up,
        ):
            inference.run(state)

        down.assert_called_once_with(button="left")
        up.assert_called_once_with(button="left")
        model.assert_called_once()
        self.assertTrue(state.stop)


if __name__ == "__main__":
    unittest.main()
