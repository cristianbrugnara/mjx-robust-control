"""
Pure JAX losses used by rollout cost terms. These functions only depend
on arrays and lightweight system geometry.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _entity_state_dim(sys: Any | None) -> int:
    if sys is None:
        return 4
    return int(getattr(sys, "entity_state_dim", 2 * int(getattr(sys, "dof_per_entity", 2))))


def _position_indices(sys: Any | None) -> tuple[int, ...]:
    if sys is None:
        return (0, 1)
    return tuple(int(i) for i in getattr(sys, "position_indices", (0, 1)))


def positions_from_state(x: jax.Array, sys: Any | None = None) -> jax.Array:
    x = jnp.asarray(x)
    entity_dim = _entity_state_dim(sys)
    pos_idx = jnp.asarray(_position_indices(sys), dtype=jnp.int32)
    x_by_entity = jnp.reshape(x, (-1, entity_dim))
    return jnp.take(x_by_entity, pos_idx, axis=-1)


def pairwise_distance_sq(q: jax.Array) -> jax.Array:
    delta = q[:, None, :] - q[None, :, :]
    return jnp.sum(delta**2, axis=-1)


def _normalize_quat(q: jax.Array) -> jax.Array:
    return q / jnp.maximum(jnp.linalg.norm(q), jnp.asarray(1.0e-8, dtype=q.dtype))


def _quat_conj(q: jax.Array) -> jax.Array:
    return jnp.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def _quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=a.dtype,
    )


def quaternion_error(q: jax.Array, target: jax.Array) -> jax.Array:
    """Return a small-angle quaternion error vector in state coordinates."""
    q = _normalize_quat(q)
    target = _normalize_quat(target)
    rel = _quat_mul(q, _quat_conj(target))
    rel = jnp.where(rel[0] < 0.0, -rel, rel)
    return jnp.array([0.0, 2.0 * rel[1], 2.0 * rel[2], 2.0 * rel[3]], dtype=q.dtype)


def state_error_with_quaternions(x: jax.Array, target: jax.Array, sys: Any | None = None) -> jax.Array:
    """Compute state error while replacing quaternion blocks by orientation error."""
    err = jnp.asarray(x) - jnp.asarray(target, dtype=jnp.asarray(x).dtype)
    if sys is None:
        return err
    blocks = tuple(getattr(sys, "quaternion_indices_per_entity", ()))
    if not blocks:
        return err
    entity_dim = _entity_state_dim(sys)
    n_agents = int(getattr(sys, "n_agents"))
    for agent in range(n_agents):
        offset = agent * entity_dim
        for block in blocks:
            idx = jnp.asarray([offset + int(i) for i in block], dtype=jnp.int32)
            q_err = quaternion_error(jnp.take(x, idx, axis=0), jnp.take(target, idx, axis=0))
            err = err.at[idx].set(q_err)
    return err


def f_loss_states(t: int | jax.Array, x: jax.Array, sys: Any, Q: jax.Array | None = None) -> jax.Array:
    """Quadratic state-tracking loss with quaternion-aware errors."""
    del t
    if Q is None:
        Q = jnp.eye(sys.n, dtype=x.dtype)
    else:
        Q = jnp.asarray(Q, dtype=x.dtype)
    xbar = jnp.asarray(sys.xbar, dtype=x.dtype)
    dx = state_error_with_quaternions(x, xbar, sys)
    return jnp.sum((Q @ dx) * dx)


def f_loss_u(t, u: jax.Array) -> jax.Array:
    """Quadratic control effort loss."""
    del t
    return jnp.sum(u**2)

def f_loss_ca(x: jax.Array, sys: Any, min_dist: float = 0.5) -> jax.Array:
    """Soft pairwise collision-avoidance loss."""
    margin = float(getattr(sys, "collision_security_margin", 0.2))
    min_sec_dist = min_dist + margin
    q = positions_from_state(x, sys)
    distance_sq = pairwise_distance_sq(q)

    n_agents = q.shape[0]
    mask = ~jnp.eye(n_agents, dtype=bool)

    active = jax.lax.stop_gradient(distance_sq) < (min_sec_dist**2)

    inv_distance = 1.0 / (distance_sq + 1e-3)
    weights = mask.astype(x.dtype) * active.astype(x.dtype)
    return 0.5 * jnp.sum(inv_distance * weights)


def f_loss_obst(x: jax.Array, sys: Any | None = None, n_agents: int = 1) -> jax.Array:
    """Soft ellipsoid obstacle loss."""
    if sys is None:
        return jnp.asarray(0.0, dtype=x.dtype)

    obstacles = tuple(getattr(sys, "obstacles", ()))
    if len(obstacles) == 0:
        return jnp.asarray(0.0, dtype=x.dtype)

    n_agents = int(getattr(sys, "n_agents", n_agents))
    q = positions_from_state(x, sys)

    centers = jnp.asarray([obs.center for obs in obstacles], dtype=x.dtype)
    radii = jnp.asarray([obs.radii for obs in obstacles], dtype=x.dtype)
    weights = jnp.asarray([getattr(obs, "weight", 1.0) for obs in obstacles], dtype=x.dtype)

    normalized = (q[:, None, :] - centers[None, :, :]) / radii[None, :, :]
    ellipsoid = jnp.sum(normalized**2, axis=-1)

    smooth_bump = jnp.exp(-ellipsoid)
    inside_penalty = jax.nn.relu(1.0 - ellipsoid) ** 2
    per_obstacle = (smooth_bump + inside_penalty) * weights[None, :]
    qq = jnp.sum(per_obstacle)

    threshold_per_agent = float(getattr(sys, "obstacle_threshold_per_agent", 0.0))
    threshold = jnp.asarray(threshold_per_agent * n_agents, dtype=x.dtype)
    return jnp.where(qq > threshold, qq, jnp.asarray(0.0, dtype=x.dtype))


def f_loss_side(x: jax.Array, sys: Any | None = None) -> jax.Array:
    """Soft box-boundary loss for agent positions."""
    if sys is None or getattr(sys, "bounds", None) is None:
        return jnp.asarray(0.0, dtype=x.dtype)

    bounds = tuple(float(v) for v in sys.bounds)
    radius = jnp.asarray(float(getattr(sys, "agent_radius", 0.0)), dtype=x.dtype)
    q = positions_from_state(x, sys)
    low = jnp.asarray(bounds[0::2], dtype=x.dtype)
    high = jnp.asarray(bounds[1::2], dtype=x.dtype)
    side = jax.nn.relu(q + radius - high) + jax.nn.relu(low + radius - q)
    return jnp.sum(side)


__all__ = [
    "positions_from_state",
    "pairwise_distance_sq",
    "f_loss_states",
    "f_loss_u",
    "f_loss_ca",
    "f_loss_obst",
    "f_loss_side",
    "quaternion_error",
    "state_error_with_quaternions",
]
