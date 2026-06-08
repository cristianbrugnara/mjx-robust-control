from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax_rollout import LossContext, RolloutConfig, _apply_qvel_impulse
from system_configs import QVelImpulseSpec
from train import sample_qvel_impulses


class DummyData(NamedTuple):
    qvel: jnp.ndarray

    def replace(self, **kwargs):
        return DummyData(qvel=kwargs.get("qvel", self.qvel))


class DummySystem:
    n_agents = 1
    qvel_dim_per_entity_resolved = 3
    qvel_impulse = QVelImpulseSpec(
        step=3,
        indices=(0, 2),
        sample_low=(-1.0, -0.25),
        sample_high=(1.0, 0.25),
        apply_to_prediction=False,
    )


class QVelImpulseTests(unittest.TestCase):
    def test_sampling_shape_and_determinism(self) -> None:
        spec = DummySystem()
        key = jax.random.PRNGKey(7)

        a = sample_qvel_impulses(key, spec, n_samples=5)
        b = sample_qvel_impulses(key, spec, n_samples=5)

        self.assertEqual(a.shape, (5, 3))
        self.assertTrue(bool(jnp.array_equal(a, b)))

        idx = jnp.asarray(spec.qvel_impulse.indices, dtype=jnp.int32)
        mask = jnp.ones((3,), dtype=bool).at[idx].set(False)
        self.assertTrue(bool(jnp.all(a[:, mask] == 0.0)))

    def test_impulse_applies_only_at_configured_step(self) -> None:
        context = LossContext(n=1, n_agents=1, xbar=jnp.zeros((1,)))
        config = RolloutConfig(
            loss_context=context,
            qpos_idx=jnp.asarray([0]),
            qvel_idx=jnp.asarray([0, 1, 2]),
            ctrl_low=jnp.zeros((1,)),
            ctrl_high=jnp.ones((1,)),
            actuator_ctrl_low=jnp.zeros((1,)),
            actuator_ctrl_high=jnp.ones((1,)),
            dof_per_entity=1,
            qvel_impulse_step=3,
        )
        data = DummyData(qvel=jnp.zeros((3,)))
        impulse = jnp.asarray([1.0, -2.0, 3.0])

        before = _apply_qvel_impulse(data, jnp.asarray(2), config, impulse)
        at_step = _apply_qvel_impulse(data, jnp.asarray(3), config, impulse)

        self.assertTrue(bool(jnp.array_equal(before.qvel, jnp.zeros((3,)))))
        self.assertTrue(bool(jnp.array_equal(at_step.qvel, impulse)))


if __name__ == "__main__":
    unittest.main()
