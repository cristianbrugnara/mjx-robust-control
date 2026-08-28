from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax_rollout


class IMCPredictionTests(unittest.TestCase):
    def test_prediction_rollout_is_one_step_from_current_real_state(self) -> None:
        source = inspect.getsource(jax_rollout._scan_step)

        self.assertIn(
            "data_prediction_next = mjx.step(mjx_model, data_real.replace(ctrl=ctrl_prediction))",
            source,
        )
        self.assertNotIn(
            "data_prediction_next = mjx.step(mjx_model, data_prediction.replace(ctrl=ctrl_prediction))",
            source,
        )


if __name__ == "__main__":
    unittest.main()
