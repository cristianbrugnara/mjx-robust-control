from __future__ import annotations

import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax_loss_functions import box_obstacle_signed_distances, f_loss_box_obst
from system_configs import BoxObstacleSpec, SystemSpec, load_system_spec
from workflow_utils import calculate_box_obstacle_violations, min_box_obstacle_margin


class DummySystem:
    n_agents = 2
    entity_state_dim = 3
    dof_per_entity = 3
    position_indices = (0, 1, 2)
    agent_radius = 0.05
    box_obstacles = (
        BoxObstacleSpec(
            center=(0.0, 0.0, 0.0),
            half_extents=(0.10, 0.10, 0.10),
            clearance=0.0,
            weight=2.0,
        ),
    )


class BoxObstacleTests(unittest.TestCase):
    def test_signed_distance_and_loss(self) -> None:
        sys_model = DummySystem()
        x = jnp.asarray([
            0.0, 0.0, 0.0,
            0.35, 0.0, 0.0,
        ])

        margins = box_obstacle_signed_distances(x, sys_model)

        self.assertEqual(margins.shape, (2, 1))
        self.assertLess(float(margins[0, 0]), 0.0)
        self.assertGreater(float(margins[1, 0]), 0.0)
        inside_loss = float(f_loss_box_obst(x, sys_model))
        self.assertGreater(inside_loss, 0.0)

        x_far = jnp.asarray([
            0.35, 0.0, 0.0,
            -0.35, 0.0, 0.0,
        ])
        self.assertAlmostEqual(float(f_loss_box_obst(x_far, sys_model)), 0.0)

    def test_smooth_box_loss_has_near_outside_gradient(self) -> None:
        sys_model = DummySystem()
        x_inside = jnp.asarray([
            0.0, 0.0, 0.0,
            0.35, 0.0, 0.0,
        ])
        x_near_outside = jnp.asarray([
            0.18, 0.0, 0.0,
            0.35, 0.0, 0.0,
        ])

        inside_loss = float(f_loss_box_obst(x_inside, sys_model))
        near_loss = float(f_loss_box_obst(x_near_outside, sys_model))
        grad = jax.grad(lambda x: f_loss_box_obst(x, sys_model))(x_near_outside)

        self.assertGreater(near_loss, 0.0)
        self.assertGreater(inside_loss, near_loss)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertGreater(float(jnp.linalg.norm(grad[:3])), 0.0)

    def test_trajectory_metrics(self) -> None:
        sys_model = DummySystem()
        traj = jnp.asarray([
            [0.35, 0.0, 0.0, -0.35, 0.0, 0.0],
            [0.00, 0.0, 0.0, -0.35, 0.0, 0.0],
        ])

        self.assertEqual(float(calculate_box_obstacle_violations(traj, sys_model)), 1.0)
        self.assertLess(float(min_box_obstacle_margin(traj, sys_model)), 0.0)

    def test_hard_config_loads_and_has_no_roof_geom(self) -> None:
        spec = load_system_spec(str(ROOT / "assets/config/crazyflies4_3d_hard.json"))

        self.assertEqual(spec.name, "crazyflies4_3d_hard")
        self.assertEqual(spec.n_agents, 4)
        self.assertTrue(spec.box_obstacles)
        self.assertEqual(spec.bounds[-1], 0.82)
        self.assertGreaterEqual(len(spec.obstacles), 6)

        centers = {tuple(round(v, 3) for v in box.center) for box in spec.box_obstacles}
        self.assertIn((0.24, 0.68, 0.28), centers)
        self.assertIn((-0.24, 0.68, 0.55), centers)
        goal_deflectors = [box for box in spec.box_obstacles if round(box.center[1], 3) == 0.68]
        self.assertEqual(len(goal_deflectors), 2)
        self.assertTrue(all(box.half_extents[1] <= 0.04 for box in goal_deflectors))

        wall_thickness = spec.box_obstacles[0].half_extents[1] * 2.0
        self.assertGreaterEqual(wall_thickness, 0.35)
        self.assertAlmostEqual(spec.box_obstacles[0].center[0] - spec.box_obstacles[0].half_extents[0], -0.95)
        self.assertAlmostEqual(spec.box_obstacles[2].center[0] + spec.box_obstacles[2].half_extents[0], 0.95)

        lower_tunnel_bottom = spec.box_obstacles[3]
        lower_tunnel_top = spec.box_obstacles[4]
        tunnel_width = lower_tunnel_bottom.half_extents[0] * 2.0
        tunnel_height = (
            lower_tunnel_top.center[2] - lower_tunnel_top.half_extents[2]
            - (lower_tunnel_bottom.center[2] + lower_tunnel_bottom.half_extents[2])
        )
        self.assertGreater(tunnel_width, 0.26)
        self.assertGreater(tunnel_height, 0.16)
        self.assertAlmostEqual(tunnel_width * tunnel_height, 0.0832, places=4)

        tree = ET.parse(ROOT / "assets/mjcf/crazyflies4_3d_hard.xml")
        geom_names = [
            geom.attrib.get("name", "").lower()
            for geom in tree.getroot().iter("geom")
        ]
        self.assertFalse(any("roof" in name or "ceiling" in name for name in geom_names))

    def test_medium_config_matches_two_way_soft_gate_layout(self) -> None:
        spec = load_system_spec(str(ROOT / "assets/config/crazyflies4_3d_medium.json"))

        self.assertEqual(spec.name, "crazyflies4_3d_medium")
        self.assertEqual(spec.n_agents, 4)
        self.assertEqual(spec.bounds[-1], 0.82)
        self.assertEqual(len(spec.obstacles), 6)
        self.assertEqual(len(spec.box_obstacles), 3)

        starts = [tuple(round(v, 3) for v in spec.x0[i:i + 3]) for i in range(0, len(spec.x0), 13)]
        goals = [tuple(round(v, 3) for v in spec.xbar[i:i + 3]) for i in range(0, len(spec.xbar), 13)]
        self.assertEqual(starts, [
            (-0.72, -0.82, 0.18),
            (0.24, -0.86, 0.30),
            (-0.24, 0.90, 0.52),
            (0.72, 0.82, 0.40),
        ])
        self.assertEqual(goals, [
            (0.72, 0.86, 0.50),
            (-0.24, 0.90, 0.34),
            (0.24, -0.86, 0.56),
            (-0.72, -0.82, 0.36),
        ])
        self.assertEqual(sum(1 for _, y, _ in starts if y < 0.0), 2)
        self.assertEqual(sum(1 for _, y, _ in starts if y > 0.0), 2)

        box_centers = {tuple(round(v, 3) for v in box.center) for box in spec.box_obstacles}
        self.assertEqual(box_centers, {
            (-0.82, 0.0, 0.43),
            (0.0, 0.0, 0.43),
            (0.82, 0.0, 0.43),
        })
        obstacle_centers = {tuple(round(v, 3) for v in obs.center) for obs in spec.obstacles}
        self.assertIn((-0.55, -0.44, 0.30), obstacle_centers)
        self.assertIn((0.55, 0.42, 0.54), obstacle_centers)

        tree = ET.parse(ROOT / "assets/mjcf/crazyflies4_3d_medium.xml")
        geom_names = [
            geom.attrib.get("name", "").lower()
            for geom in tree.getroot().iter("geom")
        ]
        self.assertTrue(any("gate_left_pillar" in name for name in geom_names))
        self.assertFalse(any("roof" in name or "ceiling" in name for name in geom_names))

    def test_invalid_box_obstacle_config_raises(self) -> None:
        with open(ROOT / "assets/config/crazyflies4_3d_hard.json", "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw["box_obstacles"][0]["half_extents"] = [0.1, -0.1, 0.1]
        with self.assertRaisesRegex(ValueError, "half_extents"):
            SystemSpec.from_dict(raw).validate_basic()


if __name__ == "__main__":
    unittest.main()
