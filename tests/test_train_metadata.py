from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from system_configs import load_system_spec
from train import TrainConfig, training_config_metadata


class TrainMetadataTests(unittest.TestCase):
    def test_sys_model_uses_resolved_system_name(self) -> None:
        config = TrainConfig(
            sys_model="corridor",
            system_config_path=str(ROOT / "assets/config/crazyflies3_3d.json"),
        )
        spec = load_system_spec(config.system_config_path)
        metadata = training_config_metadata(config, spec)

        self.assertEqual(metadata["sys_model"], "crazyflies3_3d")
        self.assertEqual(metadata["system_config_path"], config.system_config_path)


if __name__ == "__main__":
    unittest.main()
