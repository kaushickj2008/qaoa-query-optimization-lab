"""Public-demo limits, real execution, and Streamlit rendering smoke tests."""

import subprocess
import unittest
from unittest.mock import patch

from cloud_runner import RUN_LOCK, run_public, validate_public
from streamlit.testing.v1 import AppTest


class PublicDemoTests(unittest.TestCase):
    def test_limits(self):
        for config in ({"problem": "bushy4"}, {"p": 3}, {"maxiter": 101}, {"shots": 2001}):
            with self.assertRaises(ValueError):
                validate_public(config)

    def test_busy_and_timeout_release(self):
        RUN_LOCK.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "Another visitor"):
                run_public({})
        finally:
            RUN_LOCK.release()
        with patch("cloud_runner.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 120)):
            with self.assertRaisesRegex(RuntimeError, "two minutes"):
                run_public({})
        self.assertFalse(RUN_LOCK.locked())

    def test_real_run_and_ui(self):
        result = run_public({"problem": "maxcut", "p": 1, "maxiter": 10, "shots": 100})
        self.assertAlmostEqual(sum(row["probability"] for row in result["distribution"]), 1)
        self.assertEqual(sum(row["count"] for row in result["distribution"]), 100)
        self.assertEqual(result["metrics"]["exact_value"], 5)
        app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        with patch("cloud_runner.run_public", return_value=result):
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 4)


if __name__ == "__main__":
    unittest.main()
