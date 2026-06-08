from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualize import load_trajectory_set, validate_compatible_sets


class VisualizeInputTests(unittest.TestCase):
    def test_load_trajectory_set_validates_cost_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectories.npy"
            np.save(path, np.zeros((3, 5, 4), dtype=np.float32))
            np.save(path.with_name("costs.npy"), np.zeros((2,), dtype=np.float32))

            with self.assertRaises(ValueError):
                load_trajectory_set(str(path))

    def test_compare_sets_require_same_state_dim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary_path = Path(tmp) / "primary.npy"
            compare_path = Path(tmp) / "compare.npy"
            np.save(primary_path, np.zeros((3, 5, 4), dtype=np.float32))
            np.save(compare_path, np.zeros((2, 5, 6), dtype=np.float32))

            primary = load_trajectory_set(str(primary_path))
            compare = load_trajectory_set(str(compare_path))

            with self.assertRaises(ValueError):
                validate_compatible_sets(primary, compare)

    def test_compare_sets_return_shared_rollout_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary_path = Path(tmp) / "primary.npy"
            compare_path = Path(tmp) / "compare.npy"
            np.save(primary_path, np.zeros((3, 5, 4), dtype=np.float32))
            np.save(compare_path, np.zeros((2, 6, 4), dtype=np.float32))

            primary = load_trajectory_set(str(primary_path))
            compare = load_trajectory_set(str(compare_path))

            self.assertEqual(validate_compatible_sets(primary, compare), 2)


if __name__ == "__main__":
    unittest.main()
