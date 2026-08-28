from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax_rollout import LossContext, RolloutConfig, squash_or_clip_control


def _context() -> LossContext:
    return LossContext(
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


def _config(control_center, *, squash=True, margin=0.05) -> RolloutConfig:
    return RolloutConfig(
        loss_context=_context(),
        qpos_idx=jnp.arange(7),
        qvel_idx=jnp.arange(6),
        ctrl_low=jnp.asarray([0.0, -0.45, -0.45, -1.2]),
        ctrl_high=jnp.asarray([0.35, 0.45, 0.45, 1.2]),
        actuator_ctrl_low=jnp.asarray([0.0, -1.0, -1.0, -1.0]),
        actuator_ctrl_high=jnp.asarray([0.35, 1.0, 1.0, 1.0]),
        dof_per_entity=6,
        control_center=None if control_center is None else jnp.asarray(control_center),
        control_squash=squash,
        control_margin=margin,
    )


def _old_symmetric_squash(u_raw, low, high, margin):
    """Reference implementation of the previous midpoint-centered squash."""
    center = 0.5 * (low + high)
    half_width = 0.5 * (high - low) * (1.0 - margin)
    finite = jnp.isfinite(low) & jnp.isfinite(high) & (half_width > 0.0)
    squashed = center + half_width * jnp.tanh((u_raw - center) / jnp.maximum(half_width, 1.0e-6))
    return jnp.where(finite, squashed, u_raw)


class SquashCenterTests(unittest.TestCase):
    HOVER = [0.26487, 0.0, 0.0, 0.0]

    def test_control_center_is_a_fixed_point(self) -> None:
        config = _config(self.HOVER)
        c = jnp.asarray(self.HOVER)
        out = squash_or_clip_control(c, config)
        self.assertTrue(bool(jnp.allclose(out, c, atol=1e-6)), msg=f"{out} != {c}")

    def test_outputs_stay_within_bounds(self) -> None:
        config = _config(self.HOVER)
        for u in (
            jnp.asarray([1.0e3, 1.0e3, 1.0e3, 1.0e3]),
            jnp.asarray([-1.0e3, -1.0e3, -1.0e3, -1.0e3]),
        ):
            out = squash_or_clip_control(u, config)
            self.assertTrue(bool(jnp.all(out >= config.ctrl_low)))
            self.assertTrue(bool(jnp.all(out <= config.ctrl_high)))

    def test_unit_slope_at_center(self) -> None:
        config = _config(self.HOVER)
        collective = lambda u0: squash_or_clip_control(  # noqa: E731
            jnp.asarray([u0, 0.0, 0.0, 0.0]), config
        )[0]
        slope = jax.grad(collective)(0.26487)
        self.assertAlmostEqual(float(slope), 1.0, places=4)

    def test_none_reduces_to_old_symmetric_squash(self) -> None:
        config = _config(None)
        u = jnp.asarray([0.2, 0.1, -0.3, 0.7])
        out = squash_or_clip_control(u, config)
        ref = _old_symmetric_squash(u, config.ctrl_low, config.ctrl_high, config.control_margin)
        self.assertTrue(bool(jnp.allclose(out, ref, atol=1e-7)))

    def test_attitude_channels_unchanged_vs_old(self) -> None:
        config = _config(self.HOVER)
        u = jnp.asarray([0.2, 0.1, -0.3, 0.7])
        out = squash_or_clip_control(u, config)
        ref = _old_symmetric_squash(u, config.ctrl_low, config.ctrl_high, config.control_margin)
        self.assertTrue(bool(jnp.allclose(out[1:], ref[1:], atol=1e-7)))

    def test_degenerate_center_at_bound_is_nan_safe(self) -> None:
        config = _config([0.35, 0.0, 0.0, 0.0])
        fn = lambda u0: squash_or_clip_control(  # noqa: E731
            jnp.asarray([u0, 0.0, 0.0, 0.0]), config
        )[0]
        val = fn(0.4)
        grad = jax.grad(fn)(0.4)
        self.assertTrue(jnp.isfinite(val))
        self.assertTrue(jnp.isfinite(grad))
        self.assertLessEqual(float(val), 0.35 + 1e-6)

    def test_no_squash_is_plain_clip(self) -> None:
        config = _config(self.HOVER, squash=False)
        u = jnp.asarray([1.0, 1.0, -1.0, 5.0])
        out = squash_or_clip_control(u, config)
        self.assertTrue(bool(jnp.allclose(out, jnp.clip(u, config.ctrl_low, config.ctrl_high))))


if __name__ == "__main__":
    unittest.main()
