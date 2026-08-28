from __future__ import annotations

import math
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from certify import (
    dkwm_quantile_threshold,
    epsilon_m,
    k_minus,
    k_plus,
    plot_cdfs,
    plot_thresholds,
    select_examples,
    split_pass_fail,
)


class CertificationMathTests(unittest.TestCase):
    def test_epsilon_m_matches_formula(self) -> None:
        expected = math.sqrt(math.log(2.0 / 0.05) / (2.0 * 300.0))
        self.assertAlmostEqual(epsilon_m(300, 0.05), expected)

    def test_dkwm_quantile_threshold_order_statistic(self) -> None:
        values = list(range(300))
        out = dkwm_quantile_threshold(values, alpha=0.10, delta=0.05)
        expected_k_minus = math.ceil(300.0 * (1.0 - 0.10 - out.epsilon_m))
        expected_k_plus = math.ceil(300.0 * (1.0 - 0.10 + out.epsilon_m))
        self.assertEqual(out.k_minus, expected_k_minus)
        self.assertEqual(out.k_plus, expected_k_plus)
        self.assertEqual(out.lower_threshold, float(values[expected_k_minus - 1]))
        self.assertEqual(out.threshold, float(values[expected_k_plus - 1]))

    def test_order_statistic_helpers_match_formula(self) -> None:
        eps = epsilon_m(300, 0.05)
        self.assertEqual(k_minus(300, 0.10, eps), math.ceil(300.0 * (1.0 - 0.10 - eps)))
        self.assertEqual(k_plus(300, 0.10, eps), math.ceil(300.0 * (1.0 - 0.10 + eps)))

    def test_invalid_sample_size_condition_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample-size condition"):
            dkwm_quantile_threshold(range(100), alpha=0.10, delta=0.05)

    def test_nonfinite_calibration_values_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            dkwm_quantile_threshold([0.0, math.inf, 1.0], alpha=0.10, delta=0.05)

    def test_threshold_and_cdf_plots_accept_two_sided_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summaries = []
            for objective, offset in (("mean", 0.0), ("cvar", 10.0)):
                objective_dir = run_dir / objective
                objective_dir.mkdir()
                costs = np.linspace(offset, offset + 5.0, 300)
                np.save(objective_dir / "cert_costs.npy", costs)
                cert = dkwm_quantile_threshold(costs, alpha=0.10, delta=0.05)
                summaries.append(
                    {
                        "objective": objective,
                        "cert_costs_path": str(objective_dir / "cert_costs.npy"),
                        "alpha": cert.alpha,
                        "epsilon_m": cert.epsilon_m,
                        "m_cert": cert.m,
                        "k_minus": cert.k_minus,
                        "k_plus": cert.k_plus,
                        "lower_threshold": cert.lower_threshold,
                        "threshold": cert.threshold,
                    }
                )

            plot_thresholds(run_dir, summaries)
            plot_cdfs(run_dir, summaries)

            self.assertGreater((run_dir / "thresholds.png").stat().st_size, 0)
            self.assertGreater((run_dir / "cert_cdfs.png").stat().st_size, 0)
            self.assertGreater((run_dir / "mean" / "cert_cdf.png").stat().st_size, 0)
            self.assertGreater((run_dir / "cvar" / "cert_cdf.png").stat().st_size, 0)

    def test_pass_fail_split_and_selection(self) -> None:
        costs = [0.1, 0.4, 0.2, 0.9, 0.35, 1.4]
        respect, violate = split_pass_fail(costs, threshold=0.4)
        self.assertEqual(respect, [0, 1, 2, 4])
        self.assertEqual(violate, [3, 5])
        self.assertEqual(
            select_examples(respect, costs=costs, threshold=0.4, n_examples=2, prefer="respect"),
            [1, 4],
        )
        self.assertEqual(
            select_examples(violate, costs=costs, threshold=0.4, n_examples=2, prefer="violate"),
            [5, 3],
        )


if __name__ == "__main__":
    unittest.main()
