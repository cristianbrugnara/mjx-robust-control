from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
except ModuleNotFoundError: 
    mujoco = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualize import infer_agent_body_groups, infer_agent_colors_from_model


@unittest.skipIf(mujoco is None, "mujoco is not installed")
class TraceColorInferenceTests(unittest.TestCase):
    def test_freejoint_agents_are_grouped_by_qpos_slice(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/mjcf/drones3_3d.xml"))

        body_groups = infer_agent_body_groups(model, qpos_dim_per_entity=7)
        colors = infer_agent_colors_from_model(
            model,
            n_agents=3,
            qpos_dim_per_entity=7,
            alpha=0.72,
        )

        self.assertEqual(body_groups, [[1], [2], [3]])
        np.testing.assert_allclose(colors[0][:3], [0.15, 0.35, 0.78], atol=1e-6)
        np.testing.assert_allclose(colors[1][:3], [0.90, 0.40, 0.20], atol=1e-6)
        np.testing.assert_allclose(colors[2][:3], [0.22, 0.62, 0.40], atol=1e-6)

    def test_actual_car_trace_colors_ignore_taillights(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/mjcf/intersection3.xml"))

        colors = infer_agent_colors_from_model(
            model,
            n_agents=3,
            qpos_dim_per_entity=3,
            alpha=0.72,
        )

        np.testing.assert_allclose(colors[0][:3], [0.12, 0.36, 0.90], atol=1e-6)
        np.testing.assert_allclose(colors[1][:3], [0.90, 0.25, 0.18], atol=1e-6)
        np.testing.assert_allclose(colors[2][:3], [0.10, 0.62, 0.34], atol=1e-6)

    def test_identical_crazyflies_use_task_marker_colors(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/mjcf/crazyflies3_3d.xml"))

        colors = infer_agent_colors_from_model(
            model,
            n_agents=3,
            qpos_dim_per_entity=7,
            alpha=0.72,
        )

        np.testing.assert_allclose(colors[0][:3], [0.34, 0.52, 0.96], atol=1e-6)
        np.testing.assert_allclose(colors[1][:3], [0.98, 0.56, 0.30], atol=1e-6)
        np.testing.assert_allclose(colors[2][:3], [0.34, 0.84, 0.54], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
