from __future__ import annotations

from dataclasses import dataclass
import sys
import unittest
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax_rollout import extract_flat_state, state_to_qpos_qvel


@dataclass(frozen=True)
class DummyData:
    qpos: jnp.ndarray
    qvel: jnp.ndarray


class StateLayoutTests(unittest.TestCase):
    def test_unequal_qpos_qvel_state_roundtrip(self) -> None:
        x = jnp.arange(37.0)

        qpos, qvel = state_to_qpos_qvel(
            x,
            qpos_dim_per_entity=19,
            qvel_dim_per_entity=18,
        )

        self.assertEqual(qpos.shape, (19,))
        self.assertEqual(qvel.shape, (18,))

        recovered = extract_flat_state(
            DummyData(qpos=qpos, qvel=qvel),
            jnp.arange(19),
            jnp.arange(18),
            qpos_dim_per_entity=19,
            qvel_dim_per_entity=18,
        )

        self.assertTrue(bool(jnp.array_equal(recovered, x)))

    def test_equal_qpos_qvel_state_roundtrip(self) -> None:
        x = jnp.arange(12.0)

        qpos, qvel = state_to_qpos_qvel(x, dof_per_entity=3)
        recovered = extract_flat_state(
            DummyData(qpos=qpos, qvel=qvel),
            jnp.arange(6),
            jnp.arange(6),
            dof_per_entity=3,
        )

        self.assertEqual(qpos.shape, (6,))
        self.assertEqual(qvel.shape, (6,))
        self.assertTrue(bool(jnp.array_equal(recovered, x)))


if __name__ == "__main__":
    unittest.main()
