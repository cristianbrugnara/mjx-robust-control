from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax_rollout import LossContext, RolloutConfig, policy_to_actuator_control


def make_config() -> RolloutConfig:
    context = LossContext(
        n=13,
        n_agents=1,
        xbar=jnp.zeros((13,)),
        dof_per_entity=6,
        qpos_dim_per_entity=7,
        qvel_dim_per_entity=6,
        entity_state_dim=13,
        controls_per_entity=4,
        position_indices=(0, 1, 2),
        quaternion_indices_per_entity=((3, 4, 5, 6),),
    )
    return RolloutConfig(
        loss_context=context,
        qpos_idx=jnp.arange(7),
        qvel_idx=jnp.arange(6),
        ctrl_low=jnp.asarray([0.0, -0.45, -0.45, -1.2]),
        ctrl_high=jnp.asarray([0.35, 0.45, 0.45, 1.2]),
        actuator_ctrl_low=jnp.asarray([0.0, -1.0, -1.0, -1.0]),
        actuator_ctrl_high=jnp.asarray([0.35, 1.0, 1.0, 1.0]),
        dof_per_entity=6,
        control_interface_type="quadrotor_wrench_mixer",
        control_interface_params=(
            ("quaternion_indices", (3, 4, 5, 6)),
            ("omega_indices", (10, 11, 12)),
            ("roll_kp", 1.0e-5),
            ("roll_kd", 1.0e-6),
            ("pitch_kp", 1.0e-5),
            ("pitch_kd", 1.0e-6),
            ("yaw_rate_kp", 1.0e-6),
            ("moment_gears", (-1.0e-5, -1.0e-5, -1.0e-5)),
            ("moment_ctrl_limit", 1.0),
        ),
    )


class QuadrotorWrenchMixerTests(unittest.TestCase):
    def test_level_hover_maps_to_thrust_and_zero_moments(self) -> None:
        config = make_config()
        x = jnp.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u_policy = jnp.asarray([0.26487, 0.0, 0.0, 0.0])

        u_actuator = policy_to_actuator_control(x, u_policy, config)

        self.assertTrue(bool(jnp.allclose(u_actuator, jnp.asarray([0.26487, 0.0, 0.0, 0.0]))))

    def test_attitude_error_produces_finite_moment_controls(self) -> None:
        config = make_config()
        x = jnp.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u_policy = jnp.asarray([0.26487, 0.1, -0.1, 0.2])

        u_actuator = policy_to_actuator_control(x, u_policy, config)

        self.assertTrue(bool(jnp.all(jnp.isfinite(u_actuator))))
        self.assertNotEqual(float(u_actuator[1]), 0.0)
        self.assertNotEqual(float(u_actuator[2]), 0.0)
        self.assertNotEqual(float(u_actuator[3]), 0.0)

    def test_outputs_are_clipped_to_actuator_bounds(self) -> None:
        config = make_config()
        x = jnp.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u_policy = jnp.asarray([1.0, 100.0, -100.0, 100.0])

        u_actuator = policy_to_actuator_control(x, u_policy, config)

        self.assertTrue(bool(jnp.all(u_actuator >= config.actuator_ctrl_low)))
        self.assertTrue(bool(jnp.all(u_actuator <= config.actuator_ctrl_high)))


if __name__ == "__main__":
    unittest.main()
