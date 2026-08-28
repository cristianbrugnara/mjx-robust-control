from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax_rollout import LossContext, RolloutConfig, _apply_qvel_disturbance, _scan_step
from system_configs import ControllerInputSpec, QVelDisturbanceSpec
from workflow_utils import sample_qvel_disturbances


class DummyData(NamedTuple):
    qvel: jnp.ndarray

    def replace(self, **kwargs):
        return DummyData(qvel=kwargs.get("qvel", self.qvel))


class DummyMjxData(NamedTuple):
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    ctrl: jnp.ndarray

    def replace(self, **kwargs):
        return DummyMjxData(
            qpos=kwargs.get("qpos", self.qpos),
            qvel=kwargs.get("qvel", self.qvel),
            ctrl=kwargs.get("ctrl", self.ctrl),
        )


class DummyController:
    def step_from_signal(self, t, w_hat, xi):
        del t, w_hat
        return jnp.zeros((1,), dtype=xi.dtype), xi + 1.0


class DummySystem:
    n_agents = 2
    qvel_dim_per_entity_resolved = 3
    qvel_disturbance = QVelDisturbanceSpec(
        std=0.25,
        mask_per_entity=(1.0, 0.0, 2.0),
    )


class QVelDisturbanceTests(unittest.TestCase):
    def test_sampling_shape_and_determinism(self) -> None:
        spec = DummySystem()
        key = jax.random.PRNGKey(7)

        a = sample_qvel_disturbances(key, spec, n_samples=5, t_end=4)
        b = sample_qvel_disturbances(key, spec, n_samples=5, t_end=4)

        self.assertEqual(a.shape, (5, 4, 6))
        self.assertTrue(bool(jnp.array_equal(a, b)))

        self.assertTrue(bool(jnp.all(a[:, :, [1, 4]] == 0.0)))
        self.assertTrue(bool(jnp.any(a[:, :, [0, 2, 3, 5]] != 0.0)))

    def test_zero_std_produces_zero_disturbances(self) -> None:
        class ZeroDisturbanceSystem(DummySystem):
            qvel_disturbance = QVelDisturbanceSpec(
                std=0.0,
                mask_per_entity=(1.0, 1.0, 1.0),
            )

        out = sample_qvel_disturbances(
            jax.random.PRNGKey(3),
            ZeroDisturbanceSystem(),
            n_samples=2,
            t_end=5,
        )
        self.assertTrue(bool(jnp.all(out == 0.0)))

    def test_disturbance_applies_current_step(self) -> None:
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
        )
        data = DummyData(qvel=jnp.zeros((3,)))
        disturbances = jnp.asarray(
            [
                [1.0, -2.0, 3.0],
                [4.0, 5.0, -6.0],
            ]
        )

        step0 = _apply_qvel_disturbance(data, jnp.asarray(0), config, disturbances)
        step1 = _apply_qvel_disturbance(data, jnp.asarray(1), config, disturbances)

        self.assertTrue(bool(jnp.array_equal(step0.qvel, disturbances[0])))
        self.assertTrue(bool(jnp.array_equal(step1.qvel, disturbances[1])))

    def test_prediction_next_is_reanchored_to_disturbed_real_state(self) -> None:
        context = LossContext(
            n=2,
            n_agents=1,
            xbar=jnp.zeros((2,)),
            controls_per_entity=1,
        )
        config = RolloutConfig(
            loss_context=context,
            qpos_idx=jnp.asarray([0]),
            qvel_idx=jnp.asarray([0]),
            ctrl_low=jnp.asarray([-1.0]),
            ctrl_high=jnp.asarray([1.0]),
            actuator_ctrl_low=jnp.asarray([-1.0]),
            actuator_ctrl_high=jnp.asarray([1.0]),
            dof_per_entity=1,
            qpos_dim_per_entity=1,
            qvel_dim_per_entity=1,
            cost_terms=(),
            controller_inputs=(
                ControllerInputSpec(type="imc_residual", scale=1.0, clip=None, params=()),
            ),
        )
        real = DummyMjxData(
            qpos=jnp.asarray([0.0]),
            qvel=jnp.asarray([5.0]),
            ctrl=jnp.asarray([0.0]),
        )
        prediction = DummyMjxData(
            qpos=jnp.asarray([0.0]),
            qvel=jnp.asarray([100.0]),
            ctrl=jnp.asarray([0.0]),
        )
        carry = (
            real,
            prediction,
            jnp.asarray([0.0]),
            (jnp.zeros((2,)), jnp.zeros((1,))),
        )
        disturbances = jnp.asarray([[2.0]])

        def fake_step(_model, data):
            return data.replace(qvel=data.qvel + 1.0)

        with patch("jax_rollout.mjx.step", side_effect=fake_step):
            carry_next, _ = _scan_step(
                DummyController(),
                object(),
                config,
                carry,
                jnp.asarray(0),
                disturbances,
            )

        data_real_next, data_prediction_next, _, _ = carry_next
        self.assertTrue(bool(jnp.array_equal(data_real_next.qvel, jnp.asarray([8.0]))))
        self.assertTrue(bool(jnp.array_equal(data_prediction_next.qvel, jnp.asarray([8.0]))))


if __name__ == "__main__":
    unittest.main()
