from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import jax.random as jr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax_models import Controller


class ControllerInterfaceTests(unittest.TestCase):
    def test_controller_has_residual_only_interface(self) -> None:
        parameters = inspect.signature(Controller.__init__).parameters

        self.assertNotIn("f", parameters)
        self.assertFalse(hasattr(Controller, "step_from_omega"))
        self.assertFalse(hasattr(Controller, "step_from_prediction"))

    def test_controller_instance_has_no_nominal_predictor(self) -> None:
        controller = Controller(
            n=2,
            m=1,
            n_xi=3,
            l=2,
            key=jr.PRNGKey(0),
        )

        self.assertFalse(hasattr(controller, "psi_x"))
        self.assertTrue(hasattr(controller, "psi_u"))


if __name__ == "__main__":
    unittest.main()
