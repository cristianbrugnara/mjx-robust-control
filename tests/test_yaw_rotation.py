from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax_rollout import LossContext, RolloutConfig, pre_stabilizing_control

KP_XY = 0.014
KD_XY = 0.06
KP_Z = 0.18
KD_Z = 0.07
KP_YAW = 0.5
MAX_TILT = 0.5
MAX_YAW_RATE = 10.0


def _yaw_quat(psi: float):
    return [math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0)]


def _config(xbar, *, pre_stab_k: float = 0.05) -> RolloutConfig:
    context = LossContext(
        n=13,
        n_agents=1,
        xbar=jnp.asarray(xbar),
        dof_per_entity=6,
        qpos_dim_per_entity=7,
        qvel_dim_per_entity=6,
        entity_state_dim=13,
        controls_per_entity=4,
        position_indices=(0, 1, 2),
        quaternion_indices_per_entity=((3, 4, 5, 6),),
        pre_stab_mode="quadrotor_position",
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
        pre_stab_K=pre_stab_k,
        control_interface_params=(
            ("pre_stab_kp_xy", KP_XY),
            ("pre_stab_kd_xy", KD_XY),
            ("pre_stab_kp_z", KP_Z),
            ("pre_stab_kd_z", KD_Z),
            ("pre_stab_kp_yaw", KP_YAW),
            ("pre_stab_max_tilt", MAX_TILT),
            ("pre_stab_max_thrust_delta", 3.0),
            ("pre_stab_max_yaw_rate", MAX_YAW_RATE),
        ),
    )


def _state(pos, quat, vel=(0.0, 0.0, 0.0)):
    return jnp.asarray(list(pos) + list(quat) + list(vel) + [0.0, 0.0, 0.0])


class YawRotationTests(unittest.TestCase):
    GOAL = (0.3, -0.2, 0.4)
    XBAR = list(GOAL) + [1.0, 0.0, 0.0, 0.0] + [0.0] * 6

    def test_reduces_to_old_law_at_zero_yaw(self) -> None:
        config = _config(self.XBAR)
        pos = (0.0, 0.0, 0.1)
        vel = (0.05, -0.02, 0.03)
        x = _state(pos, _yaw_quat(0.0), vel)
        u = pre_stabilizing_control(x, config).reshape(4)

        e = jnp.asarray(self.GOAL) - jnp.asarray(pos)
        ax = KP_XY * e[0] - KD_XY * vel[0]
        ay = KP_XY * e[1] - KD_XY * vel[1]
        exp_collective = KP_Z * e[2] - KD_Z * vel[2]
        self.assertAlmostEqual(float(u[0]), float(exp_collective), places=6)
        self.assertAlmostEqual(float(u[2]), float(ax), places=6)
        self.assertAlmostEqual(float(u[1]), float(-ay), places=6)
        self.assertAlmostEqual(float(u[3]), 0.0, places=6)

    def test_body_frame_rotation_matches_closed_form(self) -> None:
        config = _config(self.XBAR)
        pos = (0.0, 0.0, 0.1)
        vel = (0.05, -0.02, 0.0)
        for psi in (0.3, -0.7, 1.1):
            x = _state(pos, _yaw_quat(psi), vel)
            u = pre_stabilizing_control(x, config).reshape(4)
            e = jnp.asarray(self.GOAL) - jnp.asarray(pos)
            ax = float(KP_XY * e[0] - KD_XY * vel[0])
            ay = float(KP_XY * e[1] - KD_XY * vel[1])
            c, s = math.cos(psi), math.sin(psi)
            exp_pitch = c * ax + s * ay
            exp_roll = s * ax - c * ay
            self.assertAlmostEqual(float(u[2]), exp_pitch, places=6)
            self.assertAlmostEqual(float(u[1]), exp_roll, places=6)

    def test_equilibrium_at_goal_is_zero(self) -> None:
        config = _config(self.XBAR)
        x = _state(self.GOAL, _yaw_quat(0.0), (0.0, 0.0, 0.0))
        u = pre_stabilizing_control(x, config).reshape(4)
        self.assertTrue(bool(jnp.allclose(u, jnp.zeros(4), atol=1e-7)))

    def test_yaw_regulation_is_restoring(self) -> None:
        config = _config(self.XBAR)
        x = _state(self.GOAL, _yaw_quat(0.4), (0.0, 0.0, 0.0))
        u = pre_stabilizing_control(x, config).reshape(4)
        self.assertAlmostEqual(float(u[3]), -KP_YAW * 0.4, places=5)

    def test_yaw_error_wraps_across_pi(self) -> None:
        xbar = list(self.GOAL) + _yaw_quat(0.9 * math.pi) + [0.0] * 6
        config = _config(xbar)
        x = _state(self.GOAL, _yaw_quat(-0.9 * math.pi), (0.0, 0.0, 0.0))
        u = pre_stabilizing_control(x, config).reshape(4)
        expected_rate = KP_YAW * (-0.2 * math.pi)
        self.assertAlmostEqual(float(u[3]), expected_rate, places=4)

    def test_disabled_when_pre_stab_k_zero(self) -> None:
        config = _config(self.XBAR, pre_stab_k=0.0)
        x = _state((0.0, 0.0, 0.1), _yaw_quat(0.5), (0.1, 0.1, 0.1))
        u = pre_stabilizing_control(x, config).reshape(4)
        self.assertTrue(bool(jnp.allclose(u, jnp.zeros(4))))


if __name__ == "__main__":
    unittest.main()
