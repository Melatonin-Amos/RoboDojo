"""Observation cadence needs no Isaac application or model checkpoint."""

import unittest

from env.observation_manager.obs_manager import collect_steps, control_boundary


class ObservationTimingTests(unittest.TestCase):
    def test_supported_rates(self):
        for dt, hz, steps in [(1 / 600, 30, 20), (1 / 1200, 30, 40), (.002, 25, 20), (.001, 25, 40)]:
            with self.subTest(dt=dt, hz=hz):
                self.assertEqual(collect_steps(dt, hz), steps)

    def test_float_noise_is_not_an_extra_or_missing_physics_step(self):
        self.assertEqual(collect_steps((1 / 600) * (1 + 1e-14), 30), 20)

    def test_nonintegral_and_invalid_rates_fail(self):
        for dt, hz in [(.002, 30), (.001, 30), (0., 30), (.002, float("nan"))]:
            with self.subTest(dt=dt, hz=hz), self.assertRaises(ValueError):
                collect_steps(dt, hz)

    def test_quantized_30hz_has_no_cumulative_drift(self):
        for hz in (500, 1000):
            dt = 1 / hz
            self.assertEqual(control_boundary(dt, 30, 30_000), hz * 1000)
            for index in range(1, 100):
                physical_time = control_boundary(dt, 30, index) * dt
                self.assertGreaterEqual(physical_time + 1e-12, index / 30)
                self.assertLess(physical_time - index / 30, dt)


if __name__ == "__main__":
    unittest.main()
