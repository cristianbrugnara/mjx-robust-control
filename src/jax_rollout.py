"""
Differentiable rollout engine. Defines the JAX lax.scan loop that advances MJX
physics, applies the REN controller, maps policy controls to actuators, and
turns JSON cost terms into rollout-time scalar costs. Pure loss primitives live
in jax_loss_functions.py.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from mujoco import mjx

from jax_loss_functions import (
    f_loss_ca,
    f_loss_obst,
    f_loss_side,
    f_loss_states,
    f_loss_u,
    state_error_with_quaternions,
)

Omega = tuple[jax.Array, jax.Array]
RolloutCarry = tuple[Any, Any, jax.Array, Omega]


class LossContext(eqx.Module):
    """Static/dynamic task information consumed by generic losses."""

    n: int = eqx.field(static=True)
    n_agents: int = eqx.field(static=True)
    xbar: jax.Array

    dof_per_entity: int = eqx.field(static=True, default=2)
    qpos_dim_per_entity: int = eqx.field(static=True, default=2)
    qvel_dim_per_entity: int = eqx.field(static=True, default=2)
    entity_state_dim: int = eqx.field(static=True, default=4)
    controls_per_entity: int = eqx.field(static=True, default=2)
    position_indices: tuple[int, ...] = eqx.field(static=True, default=(0, 1))
    quaternion_indices_per_entity: tuple[tuple[int, int, int, int], ...] = eqx.field(static=True, default=())

    agent_radius: float = eqx.field(static=True, default=0.0)
    collision_security_margin: float = eqx.field(static=True, default=0.2)
    bounds: tuple[float, float, float, float] | None = eqx.field(static=True, default=None)
    obstacles: tuple[Any, ...] = eqx.field(static=True, default=())
    obstacle_threshold_per_agent: float = eqx.field(static=True, default=0.0)

    pre_stab_mode: str = eqx.field(static=True, default="none")
    pre_stab_control_indices: tuple[int, ...] = eqx.field(static=True, default=())


class RolloutConfig(eqx.Module):
    loss_context: LossContext
    qpos_idx: jax.Array
    qvel_idx: jax.Array
    ctrl_low: jax.Array
    ctrl_high: jax.Array
    actuator_ctrl_low: jax.Array
    actuator_ctrl_high: jax.Array
    dof_per_entity: int = eqx.field(static=True)
    control_center: jax.Array | None = None
    control_interface_type: str = eqx.field(static=True, default="direct_actuator")
    control_interface_params: tuple[tuple[str, Any], ...] = eqx.field(static=True, default=())
    qpos_dim_per_entity: int = eqx.field(static=True, default=2)
    qvel_dim_per_entity: int = eqx.field(static=True, default=2)
    qvel_impulse_step: int | None = eqx.field(static=True, default=None)
    qvel_impulse_apply_to_prediction: bool = eqx.field(static=True, default=False)

    alpha_terminal: float = eqx.field(static=True, default=5.0)
    pre_stab_K: float = eqx.field(static=True, default=0.0)

    controller_input_clip: float = eqx.field(static=True, default=1.0)

    control_squash: bool = eqx.field(static=True, default=True)
    control_margin: float = eqx.field(static=True, default=0.05)

    Q: jax.Array | None = None

    cost_terms: tuple[Any, ...] = eqx.field(static=True, default=())
    controller_inputs: tuple[Any, ...] = eqx.field(static=True, default=())
    task_reference_names: tuple[str, ...] = eqx.field(static=True, default=())
    task_reference_values: tuple[jax.Array, ...] = ()


def state_to_qpos_qvel(
    x: jax.Array,
    *,
    dof_per_entity: int = 2,
    qpos_dim_per_entity: int | None = None,
    qvel_dim_per_entity: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Convert flat per-agent state to MuJoCo qpos/qvel."""
    x = jnp.asarray(x)
    qpos_dim = int(dof_per_entity if qpos_dim_per_entity is None else qpos_dim_per_entity)
    qvel_dim = int(dof_per_entity if qvel_dim_per_entity is None else qvel_dim_per_entity)
    entity_state_dim = qpos_dim + qvel_dim
    x_by_entity = jnp.reshape(x, (-1, entity_state_dim))
    qpos = jnp.reshape(x_by_entity[:, :qpos_dim], (-1,))
    qvel = jnp.reshape(x_by_entity[:, qpos_dim:], (-1,))
    return qpos, qvel


def extract_flat_state(
    data: Any,
    qpos_idx: jax.Array,
    qvel_idx: jax.Array,
    *,
    dof_per_entity: int = 2,
    qpos_dim_per_entity: int | None = None,
    qvel_dim_per_entity: int | None = None,
) -> jax.Array:
    """Extract flat per-agent state from MuJoCo qpos/qvel."""
    qpos_dim = int(dof_per_entity if qpos_dim_per_entity is None else qpos_dim_per_entity)
    qvel_dim = int(dof_per_entity if qvel_dim_per_entity is None else qvel_dim_per_entity)
    qpos = jnp.take(data.qpos, qpos_idx, axis=0)
    qvel = jnp.take(data.qvel, qvel_idx, axis=0)

    qpos_by_entity = jnp.reshape(qpos, (-1, qpos_dim))
    qvel_by_entity = jnp.reshape(qvel, (-1, qvel_dim))
    x = jnp.concatenate((qpos_by_entity, qvel_by_entity), axis=-1)
    return jnp.reshape(x, (-1,))


def data_with_state(
    data: Any,
    x: jax.Array,
    ctrl: jax.Array,
    *,
    dof_per_entity: int = 2,
    qpos_dim_per_entity: int | None = None,
    qvel_dim_per_entity: int | None = None,
) -> Any:
    """Return MJX data with qpos, qvel, and ctrl replaced from flat arrays."""
    qpos, qvel = state_to_qpos_qvel(
        x,
        dof_per_entity=dof_per_entity,
        qpos_dim_per_entity=qpos_dim_per_entity,
        qvel_dim_per_entity=qvel_dim_per_entity,
    )
    ctrl = jnp.asarray(ctrl, dtype=data.ctrl.dtype)
    return data.replace(
        qpos=jnp.asarray(qpos, dtype=data.qpos.dtype),
        qvel=jnp.asarray(qvel, dtype=data.qvel.dtype),
        ctrl=ctrl,
    )


def _params(spec: Any) -> dict[str, Any]:
    value = getattr(spec, "params", None)
    return {} if value is None else dict(value)


def _term_weight(spec: Any) -> float:
    return float(getattr(spec, "weight", 1.0))


def _term_where(spec: Any) -> str:
    return str(getattr(spec, "where", "running"))


def _applies(spec: Any, phase: str) -> bool:
    where = _term_where(spec)
    return where == phase or where == "both"


def _reference(
    config: RolloutConfig,
    name: str,
    dtype,
) -> jax.Array:
    if name == "xbar":
        return jnp.asarray(config.loss_context.xbar, dtype=dtype)
    for ref_name, value in zip(config.task_reference_names, config.task_reference_values):
        if ref_name == name:
            return jnp.asarray(value, dtype=dtype)
    raise ValueError(f"Task reference {name!r} was not resolved.")


def _target_array(
    config: RolloutConfig,
    target_name: str,
    dtype,
) -> jax.Array:
    return _reference(config, target_name, dtype)


def _wrap_angle_error(err: jax.Array) -> jax.Array:
    return jnp.arctan2(jnp.sin(err), jnp.cos(err))


def _state_error(
    x: jax.Array,
    target: jax.Array,
    *,
    sys: Any | None = None,
    angular_indices: Any = None,
    zero_indices: Any = None,
) -> jax.Array:
    """Compute state error with optional angle wrapping and ignored indices."""
    err = state_error_with_quaternions(x, target, sys)
    if angular_indices is not None:
        idx = jnp.asarray(angular_indices, dtype=jnp.int32)
        err = err.at[idx].set(_wrap_angle_error(jnp.take(err, idx, axis=0)))
    if zero_indices is not None:
        idx = jnp.asarray(zero_indices, dtype=jnp.int32)
        err = err.at[idx].set(jnp.zeros((idx.shape[0],), dtype=err.dtype))
    return err


def _per_entity_values(x: jax.Array, config: RolloutConfig) -> jax.Array:
    return jnp.reshape(
        x,
        (config.loss_context.n_agents, config.loss_context.entity_state_dim),
    )


def _shape_outside_distance(pos: jax.Array, shape: dict[str, Any]) -> jax.Array:
    """Distance outside one 2D road-network primitive."""
    shape_type = str(shape.get("type", "rect"))
    center = jnp.asarray(shape.get("center", (0.0, 0.0)), dtype=pos.dtype)
    if shape_type == "rect":
        half_extents = jnp.asarray(shape["half_extents"], dtype=pos.dtype)
        outside = jax.nn.relu(jnp.abs(pos - center) - half_extents)
        return jnp.linalg.norm(outside, axis=-1)
    if shape_type == "circle":
        radius = jnp.asarray(float(shape["radius"]), dtype=pos.dtype)
        return jax.nn.relu(jnp.linalg.norm(pos - center, axis=-1) - radius)
    raise ValueError(f"Unknown road_network shape type={shape_type!r}.")


def _road_network_cost(x: jax.Array, config: RolloutConfig, params: dict[str, Any]) -> jax.Array:
    """Penalty for leaving a configurable union of drivable 2D shapes."""
    shapes = tuple(dict(shape) for shape in params.get("shapes", ()))
    if not shapes:
        return jnp.zeros((), dtype=x.dtype)

    x_by_entity = _per_entity_values(x, config)
    pos_idx = jnp.asarray(config.loss_context.position_indices, dtype=jnp.int32)
    pos = jnp.take(x_by_entity, pos_idx, axis=-1)
    yaw_index = params.get("yaw_index")
    footprint = params.get("footprint_points")
    if yaw_index is not None and footprint is not None:
        yaw = x_by_entity[:, int(yaw_index)]
        offsets = jnp.asarray(footprint, dtype=x.dtype)
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        world_offsets = jnp.stack(
            [
                c[:, None] * ox[None, :] - s[:, None] * oy[None, :],
                s[:, None] * ox[None, :] + c[:, None] * oy[None, :],
            ],
            axis=-1,
        )
        pos = pos[:, None, :] + world_offsets
        pos = jnp.reshape(pos, (-1, 2))
    distances = jnp.stack([_shape_outside_distance(pos, shape) for shape in shapes], axis=-1)
    outside = jnp.min(distances, axis=-1)
    margin = jnp.asarray(float(params.get("margin", 0.0)), dtype=x.dtype)
    return jnp.sum(jax.nn.relu(outside - margin) ** 2)


def _heading_to_goal_cost(x: jax.Array, config: RolloutConfig, params: dict[str, Any]) -> jax.Array:
    """Encourage planar bodies to face their configured position targets."""
    x_by_entity = _per_entity_values(x, config)
    target_by_entity = _per_entity_values(
        jnp.asarray(config.loss_context.xbar, dtype=x.dtype),
        config,
    )
    pos_idx = jnp.asarray(config.loss_context.position_indices, dtype=jnp.int32)
    pos = jnp.take(x_by_entity, pos_idx, axis=-1)
    goal = jnp.take(target_by_entity, pos_idx, axis=-1)

    yaw_index = int(params.get("yaw_index", 2))
    yaw = x_by_entity[:, yaw_index]
    delta = goal - pos
    distance = jnp.linalg.norm(delta, axis=-1)
    desired = jnp.arctan2(delta[:, 1], delta[:, 0])
    heading_error = _wrap_angle_error(yaw - desired)
    active = (distance > float(params.get("distance_threshold", 0.4))).astype(x.dtype)
    return jnp.sum(active * (1.0 - jnp.cos(heading_error)))


def _planar_heading_velocity_cost(x: jax.Array, config: RolloutConfig, params: dict[str, Any]) -> jax.Array:
    """Soft car-like regularizer for free planar x/y/yaw bodies."""
    x_by_entity = _per_entity_values(x, config)
    yaw_index = int(params.get("yaw_index", 2))
    velocity_indices = tuple(int(i) for i in params.get("velocity_indices", (3, 4)))
    yaw = x_by_entity[:, yaw_index]
    vel = jnp.take(x_by_entity, jnp.asarray(velocity_indices, dtype=jnp.int32), axis=-1)

    forward_speed = jnp.cos(yaw) * vel[:, 0] + jnp.sin(yaw) * vel[:, 1]
    lateral_speed = -jnp.sin(yaw) * vel[:, 0] + jnp.cos(yaw) * vel[:, 1]

    lateral_weight = jnp.asarray(float(params.get("lateral_weight", 1.0)), dtype=x.dtype)
    reverse_weight = jnp.asarray(float(params.get("reverse_weight", 0.0)), dtype=x.dtype)
    min_forward_speed = jnp.asarray(float(params.get("min_forward_speed", 0.0)), dtype=x.dtype)
    total = lateral_weight * jnp.sum(lateral_speed**2)

    active = jnp.ones((x_by_entity.shape[0],), dtype=x.dtype)
    if params.get("active_distance_threshold") is not None:
        target_by_entity = _per_entity_values(
            jnp.asarray(config.loss_context.xbar, dtype=x.dtype),
            config,
        )
        pos_idx = jnp.asarray(config.loss_context.position_indices, dtype=jnp.int32)
        pos = jnp.take(x_by_entity, pos_idx, axis=-1)
        goal = jnp.take(target_by_entity, pos_idx, axis=-1)
        distance = jnp.linalg.norm(goal - pos, axis=-1)
        active = (distance > float(params["active_distance_threshold"])).astype(x.dtype)
    total = total + reverse_weight * jnp.sum(active * jax.nn.relu(min_forward_speed - forward_speed) ** 2)

    if params.get("omega_index") is not None:
        omega = x_by_entity[:, int(params["omega_index"])]
        spin_weight = jnp.asarray(float(params.get("spin_weight", 0.0)), dtype=x.dtype)
        total = total + spin_weight * jnp.sum(omega**2)
    return total


def _cost_term_value(
    spec: Any,
    t: jax.Array,
    x: jax.Array,
    u: jax.Array,
    x_prev: jax.Array,
    config: RolloutConfig,
    *,
    data: Any | None,
) -> jax.Array:
    """Evaluate one JSON-configured cost term."""
    term_type = str(getattr(spec, "type"))
    params = _params(spec)

    if term_type == "state_l2":
        target = _target_array(
            config,
            str(params.get("target", "xbar")),
            x.dtype,
        )
        q = (
            jnp.asarray(config.Q, dtype=x.dtype)
            if bool(params.get("use_Q", True)) and config.Q is not None
            else jnp.eye(x.shape[0], dtype=x.dtype)
        )
        dx = _state_error(
            x,
            target,
            sys=config.loss_context,
            angular_indices=params.get("angular_indices"),
            zero_indices=params.get("zero_indices"),
        )
        if params.get("indices") is not None:
            idx = jnp.asarray(params["indices"], dtype=jnp.int32)
            dx_idx = jnp.take(dx, idx, axis=0)
            if bool(params.get("use_Q", True)) and config.Q is not None:
                weights = jnp.take(jnp.diag(q), idx, axis=0)
                return jnp.sum(weights * dx_idx**2)
            return jnp.sum(dx_idx**2)
        return jnp.sum((q @ dx) * dx)

    if term_type == "state_bounds":
        indices = params.get("indices")
        values = x if indices is None else jnp.take(x, jnp.asarray(indices, dtype=jnp.int32), axis=0)
        low = params.get("low")
        high = params.get("high")
        margin = jnp.asarray(float(params.get("margin", 0.0)), dtype=x.dtype)
        total = jnp.zeros((), dtype=x.dtype)
        if low is not None:
            low_arr = jnp.asarray(low, dtype=x.dtype)
            total = total + jnp.sum(jax.nn.relu((low_arr + margin) - values) ** 2)
        if high is not None:
            high_arr = jnp.asarray(high, dtype=x.dtype)
            total = total + jnp.sum(jax.nn.relu(values - (high_arr - margin)) ** 2)
        return total

    if term_type == "control_l2":
        return f_loss_u(t, u)

    if term_type == "pairwise_distance_barrier":
        return f_loss_ca(x, config.loss_context, float(params.get("min_dist", 0.5)))

    if term_type == "ellipsoid_obstacle":
        return f_loss_obst(x, sys=config.loss_context)

    if term_type == "box_bounds":
        return f_loss_side(x, sys=config.loss_context)

    if term_type == "road_network":
        return _road_network_cost(x, config, params)

    if term_type == "heading_to_goal":
        return _heading_to_goal_cost(x, config, params)

    if term_type == "planar_heading_velocity":
        return _planar_heading_velocity_cost(x, config, params)

    raise ValueError(f"Unknown cost term type={term_type!r}.")


def _generic_loss(
    phase: str,
    t: jax.Array,
    x: jax.Array,
    u: jax.Array,
    x_prev: jax.Array,
    config: RolloutConfig,
    *,
    data: Any | None = None,
) -> jax.Array:
    total = jnp.zeros((), dtype=x.dtype)
    for term in config.cost_terms:
        if _applies(term, phase):
            total = total + _term_weight(term) * _cost_term_value(
                term,
                t,
                x,
                u,
                x_prev,
                config,
                data=data,
            )
    return total


def step_loss(
    t: jax.Array,
    x: jax.Array,
    u: jax.Array,
    x_prev: jax.Array,
    config: RolloutConfig,
    data: Any | None = None,
) -> jax.Array:
    return _generic_loss(
        "running",
        t,
        x,
        u,
        x_prev,
        config,
        data=data,
    )


def terminal_loss(
    t_end: int,
    x_final: jax.Array,
    config: RolloutConfig,
    data_final: Any | None = None,
) -> jax.Array:
    sys = config.loss_context
    total = jnp.zeros((), dtype=x_final.dtype)
    u_dummy = jnp.zeros_like(config.ctrl_low, dtype=x_final.dtype)
    total = total + _generic_loss(
        "terminal",
        jnp.asarray(t_end),
        x_final,
        u_dummy,
        x_final,
        config,
        data=data_final,
    )
    if config.alpha_terminal != 0.0:
        total = total + config.alpha_terminal * f_loss_states(jnp.asarray(t_end), x_final, sys, config.Q)
    return total



def pre_stabilizing_control(x: jax.Array, config: RolloutConfig) -> jax.Array:
    """Optional task-configured pre-stabilizer.

    Modes:
      - none: no pre-stabilization

      - direct_position: point-mass baseline; writes position error directly
        into force-like control channels.
        
      - quadrotor_position: simple quadrotor outer-loop baseline; writes small
        thrust/roll/pitch/yaw-rate commands in the policy space:
        [collective_thrust, roll_cmd, pitch_cmd, yaw_rate_cmd].
    """
    sys = config.loss_context
    n_ctrl = sys.n_agents * sys.controls_per_entity

    if config.pre_stab_K == 0.0 or sys.pre_stab_mode == "none":
        return jnp.zeros((n_ctrl,), dtype=x.dtype)

    xbar = jnp.asarray(sys.xbar, dtype=x.dtype)
    x_by_entity = jnp.reshape(x, (sys.n_agents, sys.entity_state_dim))
    xbar_by_entity = jnp.reshape(xbar, (sys.n_agents, sys.entity_state_dim))

    if sys.pre_stab_mode == "direct_position":
        pos_idx = jnp.asarray(sys.position_indices, dtype=jnp.int32)
        ctrl_idx = jnp.asarray(sys.pre_stab_control_indices, dtype=jnp.int32)

        pos = jnp.take(x_by_entity, pos_idx, axis=-1)
        pos_goal = jnp.take(xbar_by_entity, pos_idx, axis=-1)
        pos_error = pos_goal - pos

        u_by_entity = jnp.zeros((sys.n_agents, sys.controls_per_entity), dtype=x.dtype)
        u_by_entity = u_by_entity.at[:, ctrl_idx].set(
            jnp.asarray(config.pre_stab_K, dtype=x.dtype) * pos_error
        )
        return jnp.reshape(u_by_entity, (-1,))


    if sys.pre_stab_mode == "quadrotor_position":
        if sys.controls_per_entity != 4:
            raise ValueError("quadrotor_position pre-stabilizer expects 4 policy controls per drone.")

        params = _control_interface_params(config)

        pos = x_by_entity[:, 0:3]
        goal = xbar_by_entity[:, 0:3]
        vel = x_by_entity[:, 7:10]

        e = goal - pos

        kp_xy = jnp.asarray(float(params.get("pre_stab_kp_xy", config.pre_stab_K)), dtype=x.dtype)
        kd_xy = jnp.asarray(float(params.get("pre_stab_kd_xy", 0.35)), dtype=x.dtype)
        kp_z = jnp.asarray(float(params.get("pre_stab_kp_z", 1.20)), dtype=x.dtype)
        kd_z = jnp.asarray(float(params.get("pre_stab_kd_z", 0.70)), dtype=x.dtype)

        max_tilt = jnp.asarray(float(params.get("pre_stab_max_tilt", 0.18)), dtype=x.dtype)
        max_thrust_delta = jnp.asarray(float(params.get("pre_stab_max_thrust_delta", 3.0)), dtype=x.dtype)

        collective_delta = kp_z * e[:, 2] - kd_z * vel[:, 2]
        collective_delta = jnp.clip(collective_delta, -max_thrust_delta, max_thrust_delta)

        roll_cmd = -(kp_xy * e[:, 1] - kd_xy * vel[:, 1])
        pitch_cmd = kp_xy * e[:, 0] - kd_xy * vel[:, 0]

        roll_cmd = jnp.clip(roll_cmd, -max_tilt, max_tilt)
        pitch_cmd = jnp.clip(pitch_cmd, -max_tilt, max_tilt)

        yaw_rate_cmd = jnp.zeros_like(roll_cmd)

        u_by_entity = jnp.stack(
            [collective_delta, roll_cmd, pitch_cmd, yaw_rate_cmd],
            axis=-1,
        )
        return jnp.reshape(u_by_entity, (-1,))

    raise ValueError(f"Unknown pre_stab_mode '{sys.pre_stab_mode}'.")


def controller_feedback_signal(
    x_real: jax.Array,
    x_prediction: jax.Array,
    config: RolloutConfig,
) -> jax.Array:
    """Build the clipped REN input from JSON-configured signal blocks."""
    if config.controller_inputs:
        parts = []
        for block in config.controller_inputs:
            block_type = str(getattr(block, "type"))
            params = _params(block)
            scale = jnp.asarray(float(getattr(block, "scale", 1.0)), dtype=x_real.dtype)
            if block_type == "state":
                value = x_real
            elif block_type == "state_error":
                target = _target_array(config, str(params.get("target", "xbar")), x_real.dtype)
                sign = str(params.get("sign", "current_minus_target"))
                value = _state_error(
                    x_real,
                    target,
                    sys=config.loss_context,
                    angular_indices=params.get("angular_indices"),
                    zero_indices=params.get("zero_indices"),
                )
                if sign != "current_minus_target":
                    value = -value
            elif block_type == "imc_residual":
                value = x_real - x_prediction
            else:
                raise ValueError(f"Unknown controller input block={block_type!r}.")
            value = scale * value
            block_clip = getattr(block, "clip", None)
            if block_clip is not None:
                clip_value = jnp.asarray(float(block_clip), dtype=value.dtype)
                value = jnp.clip(value, -clip_value, clip_value)
            parts.append(value)
        w = jnp.concatenate(parts) if parts else jnp.zeros_like(x_real)
        clip = jnp.asarray(config.controller_input_clip, dtype=x_real.dtype)
        return jnp.clip(w, -clip, clip)

    raise ValueError("controller_inputs must be defined in the system JSON task config.")

def squash_or_clip_control(u_raw: jax.Array, config: RolloutConfig) -> jax.Array:
    """Keep controls in actuator ranges with useful gradients inside the range."""
    if not config.control_squash:
        return jnp.clip(u_raw, config.ctrl_low, config.ctrl_high)

    low = jnp.asarray(config.ctrl_low, dtype=u_raw.dtype)
    high = jnp.asarray(config.ctrl_high, dtype=u_raw.dtype)
    center = 0.5 * (low + high)
    half_width = 0.5 * (high - low) * (1.0 - jnp.asarray(config.control_margin, dtype=u_raw.dtype))
    finite = jnp.isfinite(low) & jnp.isfinite(high) & (half_width > 0.0)
    squashed = center + half_width * jnp.tanh((u_raw - center) / jnp.maximum(half_width, 1.0e-6))
    return jnp.where(finite, squashed, u_raw)


def _control_interface_params(config: RolloutConfig) -> dict[str, Any]:
    return dict(config.control_interface_params)


def _quat_to_roll_pitch(q: jax.Array) -> tuple[jax.Array, jax.Array]:
    q = q / jnp.maximum(jnp.linalg.norm(q), jnp.asarray(1.0e-8, dtype=q.dtype))
    qw, qx, qy, qz = q
    roll = jnp.arctan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch_sin = jnp.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = jnp.arcsin(pitch_sin)
    return roll, pitch


def _bicycle_steering_to_actuators(x: jax.Array, u_policy: jax.Array, config: RolloutConfig) -> jax.Array:
    """Map per-car policy controls to MJCF actuator controls."""
    sys = config.loss_context
    params = _control_interface_params(config)
    x_by_entity = jnp.reshape(x, (sys.n_agents, sys.entity_state_dim))
    u_by_entity = jnp.reshape(u_policy, (sys.n_agents, sys.controls_per_entity))

    yaw_index = int(params.get("yaw_index", 2))
    vx_index = int(params.get("vx_index", 3))
    vy_index = int(params.get("vy_index", 4))
    omega_index = int(params.get("omega_index", 5))
    wheelbase = jnp.asarray(float(params.get("wheelbase", 0.8)), dtype=x.dtype)
    kp = jnp.asarray(float(params.get("yaw_rate_kp", 1.8)), dtype=x.dtype)
    damping = jnp.asarray(float(params.get("yaw_rate_damping", 0.25)), dtype=x.dtype)
    torque_limit = jnp.asarray(float(params.get("yaw_torque_limit", 3.2)), dtype=x.dtype)
    lateral_damping = jnp.asarray(float(params.get("lateral_damping", 8.0)), dtype=x.dtype)
    lateral_force_limit = jnp.asarray(float(params.get("lateral_force_limit", 8.0)), dtype=x.dtype)
    actuators_per_entity = int(params.get("actuators_per_entity", 3))

    drive = u_by_entity[:, 0]
    steering = u_by_entity[:, 1]
    yaw = x_by_entity[:, yaw_index]
    vx = x_by_entity[:, vx_index]
    vy = x_by_entity[:, vy_index]
    omega = x_by_entity[:, omega_index]

    forward_speed = jnp.cos(yaw) * vx + jnp.sin(yaw) * vy
    lateral_speed = -jnp.sin(yaw) * vx + jnp.cos(yaw) * vy
    desired_yaw_rate = forward_speed / jnp.maximum(wheelbase, 1.0e-6) * jnp.tan(steering)
    yaw_torque = kp * (desired_yaw_rate - omega) - damping * omega
    yaw_torque = jnp.clip(yaw_torque, -torque_limit, torque_limit)
    if actuators_per_entity == 2:
        return jnp.reshape(jnp.stack([drive, yaw_torque], axis=-1), (-1,))
    lateral_force = jnp.clip(-lateral_damping * lateral_speed, -lateral_force_limit, lateral_force_limit)
    return jnp.reshape(jnp.stack([drive, yaw_torque, lateral_force], axis=-1), (-1,))


def _quadrotor_policy_to_rotors(x: jax.Array, u_policy: jax.Array, config: RolloutConfig) -> jax.Array:
    """Map quadrotor attitude commands to rotor thrusts."""
    sys = config.loss_context
    params = _control_interface_params(config)
    x_by_entity = jnp.reshape(x, (sys.n_agents, sys.entity_state_dim))
    u_by_entity = jnp.reshape(u_policy, (sys.n_agents, sys.controls_per_entity))

    quat_indices = tuple(params.get("quaternion_indices", (3, 4, 5, 6)))
    omega_indices = tuple(params.get("omega_indices", (10, 11, 12)))
    arm = jnp.asarray(float(params.get("arm_length", 0.32)), dtype=x.dtype)
    yaw_coeff = jnp.asarray(float(params.get("yaw_reaction_coeff", 0.035)), dtype=x.dtype)
    roll_kp = jnp.asarray(float(params.get("roll_kp", 1.4)), dtype=x.dtype)
    roll_kd = jnp.asarray(float(params.get("roll_kd", 0.45)), dtype=x.dtype)
    pitch_kp = jnp.asarray(float(params.get("pitch_kp", 1.4)), dtype=x.dtype)
    pitch_kd = jnp.asarray(float(params.get("pitch_kd", 0.45)), dtype=x.dtype)
    yaw_rate_kp = jnp.asarray(float(params.get("yaw_rate_kp", 0.30)), dtype=x.dtype)
    rotor_min = jnp.asarray(float(params.get("rotor_min", 0.0)), dtype=x.dtype)
    rotor_max = jnp.asarray(float(params.get("rotor_max", 8.0)), dtype=x.dtype)

    q_idx = jnp.asarray(quat_indices, dtype=jnp.int32)
    w_idx = jnp.asarray(omega_indices, dtype=jnp.int32)
    quat = jnp.take(x_by_entity, q_idx, axis=-1)
    omega = jnp.take(x_by_entity, w_idx, axis=-1)
    roll, pitch = jax.vmap(_quat_to_roll_pitch)(quat)

    collective = u_by_entity[:, 0]
    roll_cmd = u_by_entity[:, 1]
    pitch_cmd = u_by_entity[:, 2]
    yaw_rate_cmd = u_by_entity[:, 3]

    tau_x = roll_kp * (roll_cmd - roll) - roll_kd * omega[:, 0]
    tau_y = pitch_kp * (pitch_cmd - pitch) - pitch_kd * omega[:, 1]
    tau_z = yaw_rate_kp * (yaw_rate_cmd - omega[:, 2])

    arm_safe = jnp.maximum(arm, 1.0e-6)
    yaw_safe = jnp.maximum(yaw_coeff, 1.0e-6)
    fl = 0.25 * collective + tau_x / (4.0 * arm_safe) + tau_y / (4.0 * arm_safe) + tau_z / (4.0 * yaw_safe)
    fr = 0.25 * collective + tau_x / (4.0 * arm_safe) - tau_y / (4.0 * arm_safe) - tau_z / (4.0 * yaw_safe)
    rr = 0.25 * collective - tau_x / (4.0 * arm_safe) - tau_y / (4.0 * arm_safe) + tau_z / (4.0 * yaw_safe)
    rl = 0.25 * collective - tau_x / (4.0 * arm_safe) + tau_y / (4.0 * arm_safe) - tau_z / (4.0 * yaw_safe)
    rotors = jnp.stack([fl, fr, rr, rl], axis=-1)
    return jnp.reshape(jnp.clip(rotors, rotor_min, rotor_max), (-1,))


def _quadrotor_policy_to_wrench_actuators(x: jax.Array, u_policy: jax.Array, config: RolloutConfig) -> jax.Array:
    """Map quadrotor attitude commands to thrust plus body-moment actuators."""
    sys = config.loss_context
    params = _control_interface_params(config)
    x_by_entity = jnp.reshape(x, (sys.n_agents, sys.entity_state_dim))
    u_by_entity = jnp.reshape(u_policy, (sys.n_agents, sys.controls_per_entity))

    quat_indices = tuple(params.get("quaternion_indices", (3, 4, 5, 6)))
    omega_indices = tuple(params.get("omega_indices", (10, 11, 12)))
    roll_kp = jnp.asarray(float(params.get("roll_kp", 1.4)), dtype=x.dtype)
    roll_kd = jnp.asarray(float(params.get("roll_kd", 0.45)), dtype=x.dtype)
    pitch_kp = jnp.asarray(float(params.get("pitch_kp", 1.4)), dtype=x.dtype)
    pitch_kd = jnp.asarray(float(params.get("pitch_kd", 0.45)), dtype=x.dtype)
    yaw_rate_kp = jnp.asarray(float(params.get("yaw_rate_kp", 0.30)), dtype=x.dtype)
    moment_gears = jnp.asarray(params.get("moment_gears", (-1.0e-5, -1.0e-5, -1.0e-5)), dtype=x.dtype)
    moment_ctrl_limit = jnp.asarray(float(params.get("moment_ctrl_limit", 1.0)), dtype=x.dtype)

    q_idx = jnp.asarray(quat_indices, dtype=jnp.int32)
    w_idx = jnp.asarray(omega_indices, dtype=jnp.int32)
    quat = jnp.take(x_by_entity, q_idx, axis=-1)
    omega = jnp.take(x_by_entity, w_idx, axis=-1)
    roll, pitch = jax.vmap(_quat_to_roll_pitch)(quat)

    collective = u_by_entity[:, 0]
    roll_cmd = u_by_entity[:, 1]
    pitch_cmd = u_by_entity[:, 2]
    yaw_rate_cmd = u_by_entity[:, 3]

    tau_x = roll_kp * (roll_cmd - roll) - roll_kd * omega[:, 0]
    tau_y = pitch_kp * (pitch_cmd - pitch) - pitch_kd * omega[:, 1]
    tau_z = yaw_rate_kp * (yaw_rate_cmd - omega[:, 2])
    tau = jnp.stack([tau_x, tau_y, tau_z], axis=-1)

    gear_safe = jnp.where(
        jnp.abs(moment_gears) > 1.0e-12,
        moment_gears,
        jnp.sign(moment_gears + 1.0e-12) * 1.0e-12,
    )
    moment_ctrl = jnp.clip(tau / gear_safe[None, :], -moment_ctrl_limit, moment_ctrl_limit)
    wrench_ctrl = jnp.concatenate([collective[:, None], moment_ctrl], axis=-1)
    return jnp.reshape(wrench_ctrl, (-1,))


def policy_to_actuator_control(x: jax.Array, u_policy: jax.Array, config: RolloutConfig) -> jax.Array:
    """Map policy controls to actuator controls and clip to actuator bounds."""
    interface = config.control_interface_type
    if interface == "direct_actuator":
        u_actuator = u_policy
    elif interface == "bicycle_steering":
        u_actuator = _bicycle_steering_to_actuators(x, u_policy, config)
    elif interface == "quadrotor_attitude_mixer":
        u_actuator = _quadrotor_policy_to_rotors(x, u_policy, config)
    elif interface == "quadrotor_wrench_mixer":
        u_actuator = _quadrotor_policy_to_wrench_actuators(x, u_policy, config)
    else:
        raise ValueError(f"Unknown control_interface_type={interface!r}.")
    return jnp.clip(
        u_actuator,
        jnp.asarray(config.actuator_ctrl_low, dtype=u_actuator.dtype),
        jnp.asarray(config.actuator_ctrl_high, dtype=u_actuator.dtype),
    )


def _extract_final_state(carry: RolloutCarry, config: RolloutConfig) -> jax.Array:
    data_real, _, _, _ = carry
    return extract_flat_state(
        data_real,
        config.qpos_idx,
        config.qvel_idx,
        dof_per_entity=config.dof_per_entity,
        qpos_dim_per_entity=config.qpos_dim_per_entity,
        qvel_dim_per_entity=config.qvel_dim_per_entity,
    )


def _controller_step(controller_params: Any, t: jax.Array, w_hat: jax.Array, xi: jax.Array) -> tuple[jax.Array, jax.Array]:
    return controller_params.step_from_signal(t, w_hat, xi)


def _apply_qvel_impulse(
    data: Any,
    t: jax.Array,
    config: RolloutConfig,
    qvel_impulse: jax.Array | None,
) -> Any:
    """Apply a sampled qvel impulse at the configured rollout step."""
    if config.qvel_impulse_step is None or qvel_impulse is None:
        return data
    impulse = jnp.asarray(qvel_impulse, dtype=data.qvel.dtype)
    return jax.lax.cond(
        t == int(config.qvel_impulse_step),
        lambda d: d.replace(qvel=d.qvel + impulse),
        lambda d: d,
        data,
    )


def _scan_step(
    controller_params: Any,
    mjx_model: mjx.Model,
    loss_weights: RolloutConfig,
    carry: RolloutCarry,
    t: jax.Array,
    qvel_impulse: jax.Array | None = None,
) -> tuple[RolloutCarry, dict[str, jax.Array]]:
    """Advance one MJX step and collect rollout diagnostics."""
    data_real, data_prediction, xi, omega = carry

    data_real = _apply_qvel_impulse(data_real, t, loss_weights, qvel_impulse)
    if loss_weights.qvel_impulse_apply_to_prediction:
        data_prediction = _apply_qvel_impulse(data_prediction, t, loss_weights, qvel_impulse)

    x_real = extract_flat_state(
        data_real,
        loss_weights.qpos_idx,
        loss_weights.qvel_idx,
        dof_per_entity=loss_weights.dof_per_entity,
        qpos_dim_per_entity=loss_weights.qpos_dim_per_entity,
        qvel_dim_per_entity=loss_weights.qvel_dim_per_entity,
    )
    x_prediction = extract_flat_state(
        data_prediction,
        loss_weights.qpos_idx,
        loss_weights.qvel_idx,
        dof_per_entity=loss_weights.dof_per_entity,
        qpos_dim_per_entity=loss_weights.qpos_dim_per_entity,
        qvel_dim_per_entity=loss_weights.qvel_dim_per_entity,
    )

    x_prev, _ = omega

    w_hat = controller_feedback_signal(
        x_real,
        x_prediction,
        loss_weights,
    )
    u_boost, xi_next = _controller_step(controller_params, t, w_hat, xi)
    u_stab = pre_stabilizing_control(x_real, loss_weights)
    u_center = (
        jnp.zeros_like(u_boost)
        if loss_weights.control_center is None
        else jnp.asarray(loss_weights.control_center, dtype=u_boost.dtype)
    )
    u_policy_raw = u_center + u_boost + u_stab
    u_policy = squash_or_clip_control(u_policy_raw, loss_weights)
    u_actuator = policy_to_actuator_control(x_real, u_policy, loss_weights)
    omega_next = (x_real, u_policy)

    loss_t = step_loss(
        t,
        x_real,
        u_policy,
        x_prev,
        loss_weights,
        data=data_real,
    )

    ctrl_real = jnp.asarray(u_actuator, dtype=data_real.ctrl.dtype)
    ctrl_prediction = jnp.asarray(u_actuator, dtype=data_prediction.ctrl.dtype)

    data_real_next = mjx.step(mjx_model, data_real.replace(ctrl=ctrl_real))
    data_prediction_next = mjx.step(mjx_model, data_prediction.replace(ctrl=ctrl_prediction))

    carry_next = (data_real_next, data_prediction_next, xi_next, omega_next)
    outputs = {
        "loss": loss_t,
        "x": x_real,
        "u": u_policy,
        "u_actuator": u_actuator,
        "w": w_hat,
    }
    return carry_next, outputs


def rollout(
    controller_params: Any,
    mjx_model: mjx.Model,
    data_init: RolloutCarry,
    t_end: int,
    loss_weights: RolloutConfig,
    qvel_impulse: jax.Array | None = None,
) -> jax.Array:
    def scan_body(carry: RolloutCarry, t: jax.Array):
        return _scan_step(
            controller_params,
            mjx_model,
            loss_weights,
            carry,
            t,
            qvel_impulse,
        )

    final_carry, outputs = jax.lax.scan(scan_body, data_init, jnp.arange(t_end))
    running = jnp.mean(outputs["loss"])
    x_final = _extract_final_state(final_carry, loss_weights)
    return running + terminal_loss(
        t_end,
        x_final,
        loss_weights,
        data_final=final_carry[0],
    )


def rollout_with_trajectory(
    controller_params: Any,
    mjx_model: mjx.Model,
    data_init: RolloutCarry,
    t_end: int,
    loss_weights: RolloutConfig,
    qvel_impulse: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    def scan_body(carry: RolloutCarry, t: jax.Array):
        return _scan_step(
            controller_params,
            mjx_model,
            loss_weights,
            carry,
            t,
            qvel_impulse,
        )

    final_carry, outputs = jax.lax.scan(scan_body, data_init, jnp.arange(t_end))
    running = jnp.mean(outputs["loss"])
    x_final = _extract_final_state(final_carry, loss_weights)
    total = running + terminal_loss(
        t_end,
        x_final,
        loss_weights,
        data_final=final_carry[0],
    )
    return total, outputs["x"], outputs["u"]


__all__ = [
    "LossContext",
    "RolloutConfig",
    "RolloutCarry",
    "controller_feedback_signal",
    "data_with_state",
    "extract_flat_state",
    "policy_to_actuator_control",
    "pre_stabilizing_control",
    "_apply_qvel_impulse",
    "_quadrotor_policy_to_wrench_actuators",
    "rollout",
    "rollout_with_trajectory",
    "squash_or_clip_control",
    "state_to_qpos_qvel",
    "step_loss",
    "terminal_loss",
]
