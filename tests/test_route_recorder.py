import json
import tempfile
import unittest
from pathlib import Path

from wot_bot_v2.route_recorder import record_point, save_route


class RouteRecordingTests(unittest.TestCase):
    def test_records_only_positions_far_enough_apart(self):
        points = []
        self.assertTrue(record_point(points, (100, 100), 325, 325))
        self.assertFalse(record_point(points, (101, 100), 325, 325))
        self.assertTrue(record_point(points, (104, 100), 325, 325))
        self.assertEqual(points, [(0.3077, 0.3077), (0.32, 0.3077)])

    def test_saving_adds_route_without_changing_existing_routes(self):
        existing = {
            "berlin": {"top_spawn": [[0.2, 0.3]]},
            "moskva": {"bottom_spawn": [[0.4, 0.5]]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            path.write_text(json.dumps(existing), encoding="utf-8")

            save_route("berlin", "bottom_spawn", [(0.1, 0.8)], path)

            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["berlin"]["top_spawn"], existing["berlin"]["top_spawn"])
        self.assertEqual(saved["berlin"]["bottom_spawn"], [[0.1, 0.8]])
        self.assertEqual(saved["moskva"], existing["moskva"])

    def test_existing_route_is_not_overwritten_without_confirmation(self):
        existing = {"berlin": {"top_spawn": [[0.2, 0.3]]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            original = json.dumps(existing)
            path.write_text(original, encoding="utf-8")

            with self.assertRaises(FileExistsError):
                save_route("berlin", "top_spawn", [(0.9, 0.9)], path)

            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
