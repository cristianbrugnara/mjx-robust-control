from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import control_diagnostics, finite_quantile_summary, parse_args
import jax_rollout


def return_tuple_lengths(fn) -> set[int]:
    tree = ast.parse(inspect.getsource(fn))
    lengths: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            lengths.add(len(node.value.elts))
    return lengths


class MinimalEvaluationArtifactTests(unittest.TestCase):
    def test_timing_cli_args_parse(self) -> None:
        argv = [
            "evaluate.py",
            "--xml_path",
            "model.xml",
            "--checkpoint_path",
            "controller.eqx",
            "--timing_only",
            "--timing_repeats",
            "10",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.timing_only)
        self.assertEqual(args.timing_repeats, 10)

    def test_finite_quantile_summary_ignores_nonfinite_values(self) -> None:
        summary = finite_quantile_summary([1.0, 2.0, np.nan, 3.0])

        self.assertEqual(set(summary), {"p05", "p25", "p50", "p75", "p95"})
        self.assertAlmostEqual(summary["p50"], 2.0)

    def test_control_diagnostics_values_and_shapes(self) -> None:
        controls = np.asarray(
            [
                [
                    [0.0, 0.0],
                    [3.0, 4.0],
                    [6.0, 8.0],
                ],
                [
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, -1.0],
                ],
            ],
            dtype=np.float64,
        )

        diagnostics = control_diagnostics(controls)

        self.assertEqual(set(diagnostics), {"control_rms", "control_delta_rms", "peak_control_delta"})
        for value in diagnostics.values():
            self.assertEqual(value.shape, (2,))

        np.testing.assert_allclose(
            diagnostics["control_rms"],
            np.sqrt(np.mean(np.square(controls), axis=(1, 2))),
        )
        np.testing.assert_allclose(
            diagnostics["control_delta_rms"],
            np.asarray([np.sqrt(np.mean(np.square([[3.0, 4.0], [3.0, 4.0]]))), 0.0]),
        )
        np.testing.assert_allclose(diagnostics["peak_control_delta"], np.asarray([5.0, 0.0]))

    def test_control_diagnostics_handles_single_step(self) -> None:
        controls = np.ones((3, 1, 2), dtype=np.float32)

        diagnostics = control_diagnostics(controls)

        np.testing.assert_allclose(diagnostics["control_rms"], np.ones((3,), dtype=np.float32))
        np.testing.assert_allclose(diagnostics["control_delta_rms"], np.zeros((3,), dtype=np.float32))
        np.testing.assert_allclose(diagnostics["peak_control_delta"], np.zeros((3,), dtype=np.float32))

    def test_control_diagnostics_requires_rollout_time_control_shape(self) -> None:
        with self.assertRaises(ValueError):
            control_diagnostics(np.ones((3, 2), dtype=np.float32))

    def test_rollout_with_trajectory_interface_is_unchanged(self) -> None:
        self.assertEqual(return_tuple_lengths(jax_rollout.rollout_with_trajectory), {3})

    def test_actuator_rollout_helper_exposes_fourth_return(self) -> None:
        self.assertEqual(return_tuple_lengths(jax_rollout.rollout_with_trajectory_and_actuators), {4})
        source = inspect.getsource(jax_rollout.rollout_with_trajectory_and_actuators)
        self.assertIn('outputs["u"]', source)
        self.assertIn('outputs["u_actuator"]', source)


if __name__ == "__main__":
    unittest.main()
