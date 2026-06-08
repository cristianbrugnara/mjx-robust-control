from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import mujoco
    from mujoco import mjx
except ModuleNotFoundError:  # pragma: no cover - depends on the local env.
    mujoco = None
    mjx = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from system_configs import apply_mjx_model_options, load_system_spec


PUBLIC_SYSTEMS = (
    ("corridor", "assets/mjcf/corridor.xml", "assets/config/corridor.json"),
    ("drones3_3d", "assets/mjcf/drones3_3d.xml", "assets/config/drones3_3d.json"),
    ("crazyflies3_3d", "assets/mjcf/crazyflies3_3d.xml", "assets/config/crazyflies3_3d.json"),
    ("intersection3", "assets/mjcf/intersection3_actual_cars.xml", "assets/config/intersection3.json"),
)


@unittest.skipIf(mujoco is None, "mujoco is not installed")
class PublicSystemCompatibilityTests(unittest.TestCase):
    def test_public_systems_load_in_mjx(self) -> None:
        assert mjx is not None
        for name, xml_rel, config_rel in PUBLIC_SYSTEMS:
            with self.subTest(system=name):
                spec = load_system_spec(str(ROOT / config_rel))
                self.assertEqual(spec.name, name)
                self.assertTrue(spec.task.cost_terms)
                self.assertTrue(spec.task.controller_inputs)

                mj_model = mujoco.MjModel.from_xml_path(str(ROOT / xml_rel))
                spec.validate_against_mj_model(mj_model)
                apply_mjx_model_options(spec, mj_model)
                mjx_model = mjx.put_model(mj_model)
                _ = mjx.make_data(mjx_model)


if __name__ == "__main__":
    unittest.main()
