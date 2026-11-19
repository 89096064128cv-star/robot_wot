import math
import unittest

from aim_smoothing_demo import Point, simulate


class AimSmoothingDemoTests(unittest.TestCase):
    def test_stationary_target_is_approached_without_overshoot(self):
        samples = simulate(
            target=lambda _time: Point(10.0, 0.0),
            initial=Point(0.0, 0.0),
            max_speed=100.0,
            smoothness=5.0,
            duration=2.0,
            time_step=0.01,
        )
        distances = [math.hypot(sample.target.x - sample.follower.x,
                                sample.target.y - sample.follower.y)
                     for sample in samples]

        self.assertTrue(all(later <= earlier for earlier, later in zip(distances, distances[1:])))
        self.assertGreaterEqual(samples[-1].follower.x, 0.0)
        self.assertLessEqual(samples[-1].follower.x, 10.0)
        self.assertLess(distances[-1], 0.001)

    def test_follower_converges_while_target_moves(self):
        samples = simulate(
            target=lambda time: Point(2.0 * time, -time),
            initial=Point(-20.0, 10.0),
            max_speed=10.0,
            smoothness=8.0,
            duration=5.0,
            time_step=0.01,
        )
        initial_error = math.hypot(samples[0].target.x - samples[0].follower.x,
                                   samples[0].target.y - samples[0].follower.y)
        final_error = math.hypot(samples[-1].target.x - samples[-1].follower.x,
                                 samples[-1].target.y - samples[-1].follower.y)

        self.assertLess(final_error, initial_error)
        self.assertLess(final_error, 0.5)

    def test_higher_smoothness_tracks_stationary_target_faster(self):
        target = lambda _time: Point(10.0, 0.0)
        slow = simulate(target=target, smoothness=1.0, duration=0.5, time_step=0.01)
        fast = simulate(target=target, smoothness=8.0, duration=0.5, time_step=0.01)
        slow_error = math.hypot(slow[-1].target.x - slow[-1].follower.x,
                                slow[-1].target.y - slow[-1].follower.y)
        fast_error = math.hypot(fast[-1].target.x - fast[-1].follower.x,
                                fast[-1].target.y - fast[-1].follower.y)

        self.assertLess(fast_error, slow_error)


if __name__ == "__main__":
    unittest.main()
