from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from certify import epsilon_m, k_star, select_examples, split_pass_fail, theorem1_threshold


class CertificationMathTests(unittest.TestCase):
    def test_epsilon_m_matches_formula(self) -> None:
        expected = math.sqrt(math.log(2.0 / 0.05) / (2.0 * 300.0))
        self.assertAlmostEqual(epsilon_m(300, 0.05), expected)

    def test_theorem1_threshold_order_statistic(self) -> None:
        values = list(range(300))
        out = theorem1_threshold(values, alpha=0.10, delta=0.05)
        expected_k = math.ceil(300.0 * (1.0 - 0.10 + out.epsilon_m))
        self.assertEqual(out.k_star, expected_k)
        self.assertEqual(out.threshold, float(values[expected_k - 1]))

    def test_invalid_sample_size_condition_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample-size condition"):
            theorem1_threshold(range(100), alpha=0.10, delta=0.05)

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
