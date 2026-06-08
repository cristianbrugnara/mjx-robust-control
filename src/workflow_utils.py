"""
Setup helpers and post-rollout metrics used by train.py, evaluate.py, and certify.py.
Also contains JAX metric helpers (collision counts, obstacle margins, goal distances...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import mujoco
import numpy as np

from jax_rollout import LossContext, RolloutConfig, data_with_state
from system_configs import ControllerInputSpec, SystemSpec, TaskSpec

Array = jax.Array


def positions_from_trajectory(x: jax.Array, sys: Any | None = None) -> jax.Array:
    """Extract planar positions from trajectory tensors using a system layout.

    x:
        jax.Array with shape (..., state_dim).
    sys:
        Any object exposing ``entity_state_dim`` and ``position_indices``.
    """
    if sys is None:
        entity_state_dim = 4
        position_indices = (0, 1)
    else:
        entity_state_dim = int(getattr(sys, "entity_state_dim", 2 * int(getattr(sys, "dof_per_entity", 2))))
        position_indices = tuple(int(i) for i in getattr(sys, "position_indices", (0, 1)))

    x = jnp.asarray(x)
    leading_shape = x.shape[:-1]
    x_by_entity = jnp.reshape(x, leading_shape + (-1, entity_state_dim))
    pos_idx = jnp.asarray(position_indices, dtype=jnp.int32)
    return jnp.take(x_by_entity, pos_idx, axis=-1)


def calculate_collisions(x: jax.Array, sys: Any, min_dist: float) -> jax.Array:
    """Count unordered close-contact pairs in a trajectory batch."""
    q = positions_from_trajectory(x, sys)
    delta = q[..., :, None, :] - q[..., None, :, :]
    distance_sq = jnp.sum(delta**2, axis=-1)

    n_agents = int(getattr(sys, "n_agents"))
    eye = jnp.eye(n_agents, dtype=bool)
    mask_shape = (1,) * (distance_sq.ndim - 2) + eye.shape
    mask = jnp.reshape(~eye, mask_shape)
    collisions = (
        (distance_sq > 1.0e-4)
        & (distance_sq < (min_dist**2))
        & mask
    )
    return 0.5 * jnp.sum(collisions.astype(jnp.float32))


def _obstacle_ellipsoid_values(x: jax.Array, sys: Any) -> jax.Array:
    """Ellipsoid values for configured obstacles along a trajectory."""
    obstacles = tuple(getattr(sys, "obstacles", ()))
    q = positions_from_trajectory(x, sys)
    if len(obstacles) == 0:
        return jnp.full(q.shape[:-1] + (0,), jnp.inf, dtype=q.dtype)

    centers = jnp.asarray([obs.center for obs in obstacles], dtype=q.dtype)
    radii = jnp.asarray([obs.radii for obs in obstacles], dtype=q.dtype)
    normalized = (q[..., :, None, :] - centers) / radii
    return jnp.sum(normalized**2, axis=-1)


def calculate_obstacle_violations(x: jax.Array, sys: Any) -> jax.Array:
    """Count time/agent/obstacle samples inside configured obstacle ellipses."""
    ellipsoid = _obstacle_ellipsoid_values(x, sys)
    return jnp.sum((ellipsoid < 1.0).astype(jnp.float32))


def min_obstacle_margin(x: jax.Array, sys: Any) -> jax.Array:
    """Minimum obstacle margin along a trajectory; negative means penetration."""
    ellipsoid = _obstacle_ellipsoid_values(x, sys)
    if ellipsoid.shape[-1] == 0:
        return jnp.asarray(jnp.inf, dtype=jnp.asarray(x).dtype)
    return jnp.min(ellipsoid - 1.0)


def final_goal_distances(x: jax.Array, sys: Any, xbar: jax.Array) -> jax.Array:
    """Per-agent Euclidean distance to each goal at the final trajectory sample."""
    x = jnp.asarray(x)
    final_x = x[..., -1, :]
    q_final = positions_from_trajectory(final_x, sys)
    q_goal = positions_from_trajectory(jnp.asarray(xbar, dtype=x.dtype), sys)
    return jnp.linalg.norm(q_final - q_goal, axis=-1)


def require_explicit_task(system: SystemSpec) -> TaskSpec:
    """Return the public JSON task or fail with a clear config error."""
    if not system.task.cost_terms:
        raise ValueError(f"System '{system.name}' must define task.cost_terms in its JSON config.")
    if not system.task.controller_inputs:
        raise ValueError(f"System '{system.name}' must define task.controller_inputs in its JSON config.")
    return system.task


def controller_input_dim_from_blocks(
    *,
    state_dim: int,
    controller_inputs: tuple[ControllerInputSpec, ...],
) -> int:
    """Compute the REN input dimension from JSON controller-input blocks."""
    total = 0
    for block in controller_inputs:
        if block.type in ("state", "state_error", "imc_residual"):
            total += int(state_dim)
        else:
            raise ValueError(f"Unknown controller input block {block.type!r}.")
    return int(total)


def actuator_ctrl_bounds(mj_model: mujoco.MjModel, *, dtype: Any = jnp.float32) -> tuple[Array, Array]:
    """Return per-actuator bounds, using wide finite bounds for unlimited actuators."""
    ctrlrange = np.asarray(mj_model.actuator_ctrlrange, dtype=np.float32)
    limited = np.asarray(mj_model.actuator_ctrllimited, dtype=bool)
    low = np.where(limited, ctrlrange[:, 0], -1.0e6)
    high = np.where(limited, ctrlrange[:, 1], 1.0e6)
    return jnp.asarray(low, dtype=dtype), jnp.asarray(high, dtype=dtype)


def policy_ctrl_bounds(
    system: SystemSpec,
    actuator_low: Array,
    actuator_high: Array,
    *,
    dtype: Any = jnp.float32,
) -> tuple[Array, Array]:
    """Return policy-space bounds, falling back to actuator bounds."""
    if system.policy_control_low and system.policy_control_high:
        return (
            jnp.asarray(system.policy_control_low, dtype=dtype),
            jnp.asarray(system.policy_control_high, dtype=dtype),
        )
    return jnp.asarray(actuator_low, dtype=dtype), jnp.asarray(actuator_high, dtype=dtype)


def resolve_task_references(
    task: TaskSpec,
    *,
    base_dir: Path,
    dtype: Any = jnp.float32,
) -> tuple[tuple[str, Array], ...]:
    """Load task references declared inline or as sidecar arrays."""
    refs: list[tuple[str, Array]] = []
    for ref in task.references:
        if ref.path is not None:
            path = Path(ref.path)
            if not path.is_absolute():
                path = base_dir / path
            value = np.load(path)
        else:
            value = ref.value
        refs.append((ref.name, jnp.asarray(value, dtype=dtype)))
    return tuple(refs)


def build_loss_context(*, system: SystemSpec, xbar: Array) -> LossContext:
    """Build the static loss context from a system spec."""
    return LossContext(
        n=int(xbar.shape[0]),
        n_agents=system.n_agents,
        dof_per_entity=system.dof_per_entity,
        qpos_dim_per_entity=system.qpos_dim_per_entity_resolved,
        qvel_dim_per_entity=system.qvel_dim_per_entity_resolved,
        entity_state_dim=system.entity_state_dim,
        controls_per_entity=system.controls_per_entity,
        position_indices=system.position_indices,
        quaternion_indices_per_entity=system.quaternion_indices_per_entity,
        agent_radius=system.agent_radius,
        collision_security_margin=system.collision_security_margin,
        bounds=system.bounds,
        obstacles=system.obstacles,
        obstacle_threshold_per_agent=system.obstacle_threshold_per_agent,
        pre_stab_mode=system.pre_stab_mode,
        pre_stab_control_indices=system.pre_stab_control_indices,
        xbar=xbar,
    )


def build_rollout_config(
    *,
    system: SystemSpec,
    xbar: Array,
    Q: Array,
    qpos_idx: Array,
    qvel_idx: Array,
    pre_stab_K: float,
    alpha_terminal: float,
    controller_input_clip: float,
    control_squash: bool,
    control_margin: float,
    ctrl_low: Array,
    ctrl_high: Array,
    actuator_ctrl_low: Array,
    actuator_ctrl_high: Array,
    task: TaskSpec,
    task_references: tuple[tuple[str, Array], ...],
) -> RolloutConfig:
    """Build the rollout configuration passed to JAX/MJX scans."""
    return RolloutConfig(
        loss_context=build_loss_context(system=system, xbar=xbar),
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        ctrl_low=ctrl_low,
        ctrl_high=ctrl_high,
        actuator_ctrl_low=actuator_ctrl_low,
        actuator_ctrl_high=actuator_ctrl_high,
        control_center=(
            jnp.asarray(system.control_center, dtype=xbar.dtype)
            if system.control_center else None
        ),
        control_interface_type=system.control_interface.type,
        control_interface_params=tuple(system.control_interface.params),
        dof_per_entity=system.dof_per_entity,
        qpos_dim_per_entity=system.qpos_dim_per_entity_resolved,
        qvel_dim_per_entity=system.qvel_dim_per_entity_resolved,
        qvel_impulse_step=(
            int(system.qvel_impulse.step) if system.qvel_impulse is not None else None
        ),
        qvel_impulse_apply_to_prediction=(
            bool(system.qvel_impulse.apply_to_prediction)
            if system.qvel_impulse is not None else False
        ),
        alpha_terminal=alpha_terminal,
        pre_stab_K=pre_stab_K,
        controller_input_clip=controller_input_clip,
        control_squash=control_squash,
        control_margin=control_margin,
        Q=Q,
        cost_terms=task.cost_terms,
        controller_inputs=task.controller_inputs,
        task_reference_names=tuple(name for name, _ in task_references),
        task_reference_values=tuple(value for _, value in task_references),
    )


def sample_initial_conditions(
    key: Array,
    x0: Array,
    *,
    std_ini: float,
    n_samples: int,
    init_noise_mask: Array,
    n_agents: int | None = None,
    entity_state_dim: int | None = None,
    quaternion_indices_per_entity: tuple[tuple[int, int, int, int], ...] = (),
) -> Array:
    """Sample noisy initial states and renormalize quaternion blocks."""
    noise = jr.normal(key, (n_samples, x0.shape[0]), dtype=x0.dtype)
    samples = x0[None, :] + std_ini * noise * init_noise_mask[None, :]
    if not quaternion_indices_per_entity:
        return samples
    if n_agents is None or entity_state_dim is None:
        raise ValueError("Quaternion initial condition normalization requires n_agents and entity_state_dim.")
    for agent in range(int(n_agents)):
        offset = agent * int(entity_state_dim)
        for block in quaternion_indices_per_entity:
            idx = jnp.asarray([offset + int(i) for i in block], dtype=jnp.int32)
            quat = jnp.take(samples, idx, axis=1)
            quat = quat / jnp.maximum(
                jnp.linalg.norm(quat, axis=1, keepdims=True),
                jnp.asarray(1.0e-8, dtype=samples.dtype),
            )
            samples = samples.at[:, idx].set(quat)
    return samples


def sample_qvel_impulses(
    key: Array,
    spec: SystemSpec,
    *,
    n_samples: int,
    dtype: Any = jnp.float32,
) -> Array:
    """Draw full-nv qvel impulse vectors, zero outside configured indices."""
    qvel_dim = spec.n_agents * spec.qvel_dim_per_entity_resolved
    if spec.qvel_impulse is None:
        return jnp.zeros((n_samples, qvel_dim), dtype=dtype)
    impulse = spec.qvel_impulse
    low = jnp.asarray(impulse.sample_low, dtype=dtype)
    high = jnp.asarray(impulse.sample_high, dtype=dtype)
    uniform = jr.uniform(key, (n_samples, low.shape[0]), dtype=dtype)
    sampled = low[None, :] + uniform * (high - low)[None, :]
    out = jnp.zeros((n_samples, qvel_dim), dtype=dtype)
    idx = jnp.asarray(impulse.indices, dtype=jnp.int32)
    return out.at[:, idx].set(sampled)


def build_data_init(
    data_template: Any,
    x_real0: Array,
    x_prediction0: Array,
    *,
    xi_dim: int,
    ctrl_dim: int,
    policy_ctrl_dim: int | None = None,
    dof_per_entity: int,
    qpos_dim_per_entity: int | None = None,
    qvel_dim_per_entity: int | None = None,
) -> tuple[Any, Any, Array, tuple[Array, Array]]:
    """Create the initial scan carry for real and predicted MJX states."""
    u0 = jnp.zeros((ctrl_dim,), dtype=x_real0.dtype)
    data_real0 = data_with_state(
        data_template,
        x_real0,
        u0,
        dof_per_entity=dof_per_entity,
        qpos_dim_per_entity=qpos_dim_per_entity,
        qvel_dim_per_entity=qvel_dim_per_entity,
    )
    data_prediction0 = data_with_state(
        data_template,
        x_prediction0,
        u0,
        dof_per_entity=dof_per_entity,
        qpos_dim_per_entity=qpos_dim_per_entity,
        qvel_dim_per_entity=qvel_dim_per_entity,
    )
    xi0 = jnp.zeros((xi_dim,), dtype=x_real0.dtype)
    u_policy0 = jnp.zeros((ctrl_dim if policy_ctrl_dim is None else policy_ctrl_dim,), dtype=x_real0.dtype)
    omega0 = (x_real0, u_policy0)
    return (data_real0, data_prediction0, xi0, omega0)
