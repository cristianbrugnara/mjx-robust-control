"""
Evaluation workflow. Loads a trained checkpoint, runs held-out MJX rollouts, and writes
trajectories, controls, costs, and a JSON summary to disk.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mujoco
import numpy as np
from mujoco import mjx

from jax_models import Controller
from jax_rollout import (
    RolloutConfig,
    rollout_with_trajectory,
)
from system_configs import (
    SystemSpec,
    TaskSpec,
    apply_mjx_model_options,
    load_system_spec,
)
from workflow_utils import (
    actuator_ctrl_bounds,
    build_data_init,
    build_rollout_config as build_shared_rollout_config,
    calculate_collisions,
    calculate_obstacle_violations,
    controller_input_dim_from_blocks,
    final_goal_distances,
    min_obstacle_margin,
    policy_ctrl_bounds,
    require_explicit_task,
    resolve_task_references,
    sample_initial_conditions,
    sample_qvel_impulses,
)



@dataclass(frozen=True)
class EvalSpec:
    system: SystemSpec
    x0: jax.Array
    xbar: jax.Array
    Q: jax.Array
    t_end: int
    n_xi: int
    l: int
    std_ini: float
    init_noise_mask: jax.Array
    qpos_idx: jax.Array
    qvel_idx: jax.Array
    pre_stab_K: float
    use_sp: bool
    std_ini_param: float
    t_end_sp: int | None
    output_amplification: float
    psi_u_inner_output_gain: float
    controller_input_clip: float
    control_squash: bool
    control_margin: float
    ctrl_low: jax.Array
    ctrl_high: jax.Array
    actuator_ctrl_low: jax.Array
    actuator_ctrl_high: jax.Array
    alpha_terminal: float
    controller_input_dim: int
    task: TaskSpec = field(default_factory=TaskSpec)
    task_references: tuple[tuple[str, jax.Array], ...] = ()

    @property
    def n_agents(self) -> int:
        return self.system.n_agents

    @property
    def dof_per_entity(self) -> int:
        return self.system.dof_per_entity

    @property
    def qpos_dim_per_entity(self) -> int:
        return self.system.qpos_dim_per_entity_resolved

    @property
    def qvel_dim_per_entity(self) -> int:
        return self.system.qvel_dim_per_entity_resolved


class _CollisionContext:
    def __init__(self, system: SystemSpec) -> None:
        self.n_agents = system.n_agents
        self.dof_per_entity = system.dof_per_entity
        self.entity_state_dim = system.entity_state_dim
        self.position_indices = system.position_indices


def zero_nominal_prediction(t: int | jax.Array, y: jax.Array, u: jax.Array) -> jax.Array:
    del t, u
    return y


def _as_array(value: Any, *, dtype: Any = jnp.float32) -> jax.Array:
    return jnp.asarray(value, dtype=dtype)


def _rollout_meta(raw: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("resolved", "rollout"):
        value = raw.get(key, {})
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _build_spec_from_metadata(raw: dict[str, Any], mj_model: mujoco.MjModel, sys_model: str | None) -> EvalSpec:
    del sys_model
    if "system_config" not in raw:
        raise ValueError(
            "Checkpoint metadata does not contain system_config. "
            "Old checkpoint compatibility has been removed; retrain with the public workflow."
        )
    system = SystemSpec.from_dict(raw["system_config"])

    task_meta = raw.get("task", {})
    ctrl = raw["controller"]
    resolved = raw.get("resolved", {})
    rollout_meta = _rollout_meta(raw)
    actuator_low_default, actuator_high_default = actuator_ctrl_bounds(mj_model)
    ctrl_low_default, ctrl_high_default = policy_ctrl_bounds(
        system,
        actuator_low_default,
        actuator_high_default,
    )

    x0 = _as_array(task_meta.get("x0", system.x0))
    xbar = _as_array(task_meta.get("xbar", system.xbar))
    Q = _as_array(task_meta.get("Q", system.Q().tolist()))
    qpos_idx = jnp.asarray(
        task_meta.get("qpos_idx", system.resolve_qpos_idx(mj_model.nq)),
        dtype=jnp.int32,
    )
    qvel_idx = jnp.asarray(
        task_meta.get("qvel_idx", system.resolve_qvel_idx(mj_model.nv)),
        dtype=jnp.int32,
    )
    has_serialized_task = isinstance(rollout_meta.get("task"), dict)
    task = TaskSpec.from_dict(rollout_meta["task"]) if has_serialized_task else require_explicit_task(system)
    if not task.controller_inputs:
        raise ValueError(f"System '{system.name}' task must define controller_inputs.")
    if "task_references" in rollout_meta:
        task_references = tuple(
            (str(name), jnp.asarray(value, dtype=x0.dtype))
            for name, value in rollout_meta["task_references"]
        )
    else:
        task_references = resolve_task_references(task, base_dir=Path.cwd(), dtype=x0.dtype)
    input_dim = int(
        ctrl.get(
            "n",
            rollout_meta.get(
                "controller_input_dim",
                controller_input_dim_from_blocks(
                    state_dim=int(xbar.shape[0]),
                    controller_inputs=task.controller_inputs,
                ),
            ),
        )
    )

    return EvalSpec(
        system=system,
        x0=x0,
        xbar=xbar,
        Q=Q,
        t_end=int(task_meta.get("t_end", system.t_end)),
        n_xi=int(ctrl.get("n_xi", system.n_xi)),
        l=int(ctrl.get("l", system.l)),
        std_ini=float(task_meta.get("std_ini", resolved.get("std_ini", system.std_ini))),
        init_noise_mask=_as_array(task_meta.get("init_noise_mask", system.init_noise_mask().tolist())),
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        pre_stab_K=float(task_meta.get("pre_stab_K", resolved.get("pre_stab_K", system.pre_stab_K))),
        use_sp=bool(ctrl.get("use_sp", resolved.get("use_sp", system.use_sp))),
        std_ini_param=float(ctrl.get("std_ini_param", resolved.get("std_ini_param", system.std_ini_param))),
        t_end_sp=int(ctrl.get("t_end_sp", system.t_end)),
        output_amplification=float(
            ctrl.get("output_amplification", resolved.get("output_amplification", system.output_amplification))
        ),
        psi_u_inner_output_gain=float(
            ctrl.get(
                "psi_u_inner_output_gain",
                resolved.get("psi_u_inner_output_gain", system.psi_u_inner_output_gain),
            )
        ),
        controller_input_clip=float(rollout_meta.get("controller_input_clip", 1.0)),
        control_squash=bool(rollout_meta.get("control_squash", True)),
        control_margin=float(rollout_meta.get("control_margin", 0.05)),
        ctrl_low=_as_array(rollout_meta.get("ctrl_low", ctrl_low_default.tolist())),
        ctrl_high=_as_array(rollout_meta.get("ctrl_high", ctrl_high_default.tolist())),
        actuator_ctrl_low=_as_array(
            rollout_meta.get("actuator_ctrl_low", actuator_low_default.tolist())
        ),
        actuator_ctrl_high=_as_array(
            rollout_meta.get("actuator_ctrl_high", actuator_high_default.tolist())
        ),
        alpha_terminal=float(rollout_meta.get("alpha_terminal", 0.0)),
        controller_input_dim=input_dim,
        task=task,
        task_references=task_references,
    )


def _build_spec_from_sys_model(sys_model: str, mj_model: mujoco.MjModel) -> EvalSpec:
    system = load_system_spec(sys_model)
    system.validate_against_mj_model(mj_model)
    task = require_explicit_task(system)
    actuator_ctrl_low, actuator_ctrl_high = actuator_ctrl_bounds(mj_model)
    ctrl_low, ctrl_high = policy_ctrl_bounds(system, actuator_ctrl_low, actuator_ctrl_high)
    return EvalSpec(
        system=system,
        x0=system.x0_array(),
        xbar=system.xbar_array(),
        Q=system.Q(),
        t_end=int(system.t_end),
        n_xi=int(system.n_xi),
        l=int(system.l),
        std_ini=float(system.std_ini),
        init_noise_mask=system.init_noise_mask(),
        qpos_idx=jnp.asarray(system.resolve_qpos_idx(mj_model.nq), dtype=jnp.int32),
        qvel_idx=jnp.asarray(system.resolve_qvel_idx(mj_model.nv), dtype=jnp.int32),
        pre_stab_K=float(system.pre_stab_K),
        use_sp=bool(system.use_sp),
        std_ini_param=float(system.std_ini_param),
        t_end_sp=int(system.t_end),
        output_amplification=float(system.output_amplification),
        psi_u_inner_output_gain=float(system.psi_u_inner_output_gain),
        controller_input_clip=1.0,
        control_squash=True,
        control_margin=0.05,
        ctrl_low=ctrl_low,
        ctrl_high=ctrl_high,
        actuator_ctrl_low=actuator_ctrl_low,
        actuator_ctrl_high=actuator_ctrl_high,
        alpha_terminal=0.0,
        controller_input_dim=controller_input_dim_from_blocks(
            state_dim=int(system.state_dim),
            controller_inputs=task.controller_inputs,
        ),
        task=task,
        task_references=(),
    )


def load_eval_spec(
    *,
    checkpoint_path: str,
    sys_model: str | None,
    system_config_path: str | None,
    mj_model: mujoco.MjModel,
) -> EvalSpec:
    """Load evaluation settings from checkpoint metadata or a system JSON.

    A checkpoint sidecar fixes the task, controller dimensions, bounds, and
    rollout knobs used at training time. Without a sidecar, pass a system config
    or system name so evaluation can rebuild the same public workflow settings.
    """
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    if meta_path.exists():
        if system_config_path is not None:
            warnings.warn(
                "Checkpoint metadata found; evaluation will use the task settings saved with the checkpoint "
                "instead of --system_config_path. Train a new checkpoint to pick up config changes.",
                stacklevel=2,
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            spec = _build_spec_from_metadata(json.load(f), mj_model, sys_model)
        spec.system.validate_against_mj_model(mj_model)
        return spec

    if system_config_path is not None:
        system = load_system_spec(system_config_path)
        system.validate_against_mj_model(mj_model)
        task = require_explicit_task(system)
        task_references = resolve_task_references(
            task, base_dir=Path(system_config_path).resolve().parent
        )
        actuator_ctrl_low, actuator_ctrl_high = actuator_ctrl_bounds(mj_model)
        ctrl_low, ctrl_high = policy_ctrl_bounds(system, actuator_ctrl_low, actuator_ctrl_high)
        return EvalSpec(
            system=system,
            x0=system.x0_array(),
            xbar=system.xbar_array(),
            Q=system.Q(),
            t_end=int(system.t_end),
            n_xi=int(system.n_xi),
            l=int(system.l),
            std_ini=float(system.std_ini),
            init_noise_mask=system.init_noise_mask(),
            qpos_idx=jnp.asarray(system.resolve_qpos_idx(mj_model.nq), dtype=jnp.int32),
            qvel_idx=jnp.asarray(system.resolve_qvel_idx(mj_model.nv), dtype=jnp.int32),
            pre_stab_K=float(system.pre_stab_K),
            use_sp=bool(system.use_sp),
            std_ini_param=float(system.std_ini_param),
            t_end_sp=int(system.t_end),
            output_amplification=float(system.output_amplification),
            psi_u_inner_output_gain=float(system.psi_u_inner_output_gain),
            controller_input_clip=1.0,
            control_squash=True,
            control_margin=0.05,
            ctrl_low=ctrl_low,
            ctrl_high=ctrl_high,
            actuator_ctrl_low=actuator_ctrl_low,
            actuator_ctrl_high=actuator_ctrl_high,
            alpha_terminal=0.0,
            controller_input_dim=controller_input_dim_from_blocks(
                state_dim=int(system.state_dim),
                controller_inputs=task.controller_inputs,
            ),
            task=task,
            task_references=task_references,
        )

    if sys_model is None:
        raise ValueError("No checkpoint metadata found. Provide --sys_model or --system_config_path.")
    return _build_spec_from_sys_model(sys_model, mj_model)


def build_rollout_config(spec: EvalSpec) -> RolloutConfig:
    return build_shared_rollout_config(
        system=spec.system,
        xbar=spec.xbar,
        Q=spec.Q,
        qpos_idx=spec.qpos_idx,
        qvel_idx=spec.qvel_idx,
        pre_stab_K=spec.pre_stab_K,
        alpha_terminal=spec.alpha_terminal,
        controller_input_clip=spec.controller_input_clip,
        control_squash=spec.control_squash,
        control_margin=spec.control_margin,
        ctrl_low=spec.ctrl_low,
        ctrl_high=spec.ctrl_high,
        actuator_ctrl_low=spec.actuator_ctrl_low,
        actuator_ctrl_high=spec.actuator_ctrl_high,
        task=spec.task,
        task_references=spec.task_references,
    )


def build_controller_skeleton(spec: EvalSpec, mj_model: mujoco.MjModel) -> Controller:
    del mj_model
    return Controller(
        zero_nominal_prediction,
        n=int(spec.controller_input_dim),
        m=int(spec.system.control_dim),
        n_xi=int(spec.n_xi),
        l=int(spec.l),
        key=jr.PRNGKey(0),
        use_sp=spec.use_sp,
        t_end_sp=spec.t_end_sp,
        std_ini_param=spec.std_ini_param,
        output_amplification=spec.output_amplification,
        psi_u_inner_output_gain=spec.psi_u_inner_output_gain,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--n_rollouts", type=int, default=100)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Evaluate rollouts in chunks to reduce peak GPU memory. Defaults to all rollouts at once.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--sys_model", type=str, default=None)
    parser.add_argument("--system_config_path", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--controller_input_clip", type=float, default=None)
    parser.set_defaults(control_squash=None)
    parser.add_argument("--control_squash", dest="control_squash", action="store_true")
    parser.add_argument("--no_control_squash", dest="control_squash", action="store_false")
    parser.add_argument("--control_margin", type=float, default=None)
    parser.add_argument("--alpha_terminal", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_stem = Path(args.checkpoint_path).stem
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path("eval_results") / ckpt_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    mj_model = mujoco.MjModel.from_xml_path(args.xml_path)
    spec = load_eval_spec(
        checkpoint_path=args.checkpoint_path,
        sys_model=args.sys_model,
        system_config_path=args.system_config_path,
        mj_model=mj_model,
    )
    apply_mjx_model_options(spec.system, mj_model)
    mjx_model = mjx.put_model(mj_model)
    data_template = mjx.make_data(mjx_model)
    override_kwargs = {}
    for name in (
        "controller_input_clip",
        "control_squash",
        "control_margin",
        "alpha_terminal",
    ):
        value = getattr(args, name, None)
        if value is not None:
            override_kwargs[name] = value
    if override_kwargs:
        spec = replace(spec, **override_kwargs)
    rollout_config = build_rollout_config(spec)

    controller_skeleton = build_controller_skeleton(spec, mj_model)
    controller = eqx.tree_deserialise_leaves(args.checkpoint_path, controller_skeleton)

    key = jr.PRNGKey(args.seed)
    key_x0, key_impulse = jr.split(key, 2)
    x0_batch = sample_initial_conditions(
        key_x0,
        spec.x0,
        std_ini=spec.std_ini,
        n_samples=args.n_rollouts,
        init_noise_mask=spec.init_noise_mask,
        n_agents=spec.system.n_agents,
        entity_state_dim=spec.system.entity_state_dim,
        quaternion_indices_per_entity=spec.system.quaternion_indices_per_entity,
    )
    qvel_impulse_batch = sample_qvel_impulses(
        key_impulse,
        spec.system,
        n_samples=args.n_rollouts,
        dtype=spec.x0.dtype,
    )

    x_prediction0 = spec.xbar
    xi_dim = controller.psi_u.n_xi
    ctrl_dim = int(mj_model.nu)

    def single_eval(
        x_real0: jax.Array,
        qvel_impulse: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        data_init = build_data_init(
            data_template,
            x_real0,
            x_prediction0,
            xi_dim=xi_dim,
            ctrl_dim=ctrl_dim,
            policy_ctrl_dim=int(spec.system.control_dim),
            dof_per_entity=spec.system.dof_per_entity,
            qpos_dim_per_entity=spec.system.qpos_dim_per_entity_resolved,
            qvel_dim_per_entity=spec.system.qvel_dim_per_entity_resolved,
        )
        return rollout_with_trajectory(
            controller,
            mjx_model,
            data_init,
            spec.t_end,
            rollout_config,
            qvel_impulse,
        )

    batched_eval = jax.jit(jax.vmap(single_eval))
    eval_batch_size = int(args.eval_batch_size or args.n_rollouts)
    if eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be positive.")

    cost_batches = []
    trajectory_batches = []
    control_batches = []
    for start in range(0, args.n_rollouts, eval_batch_size):
        end = min(start + eval_batch_size, args.n_rollouts)
        costs_i, trajectories_i, controls_i = batched_eval(
            x0_batch[start:end],
            qvel_impulse_batch[start:end],
        )
        cost_batches.append(np.asarray(costs_i))
        trajectory_batches.append(np.asarray(trajectories_i))
        control_batches.append(np.asarray(controls_i))

    costs = np.concatenate(cost_batches, axis=0)
    trajectories = np.concatenate(trajectory_batches, axis=0)
    controls = np.concatenate(control_batches, axis=0)

    trajectories_np = trajectories
    controls_np = controls
    costs_np = costs

    np.save(output_dir / "trajectories.npy", trajectories_np)
    np.save(output_dir / "controls.npy", controls_np)
    np.save(output_dir / "costs.npy", costs_np)
    np.save(output_dir / "initial_states.npy", np.asarray(x0_batch))
    np.save(output_dir / "disturbances.npy", np.asarray(qvel_impulse_batch))

    collision_ctx = _CollisionContext(spec.system)
    per_rollout_collisions = np.asarray(
        jax.jit(
            jax.vmap(
                lambda tr: calculate_collisions(tr[None, ...], collision_ctx, spec.system.min_dist)
            )
        )(trajectories)
    )
    per_rollout_obstacle_violations = np.asarray(
        jax.jit(jax.vmap(lambda tr: calculate_obstacle_violations(tr, spec.system)))(trajectories)
    )
    per_rollout_min_obstacle_margin = np.asarray(
        jax.jit(jax.vmap(lambda tr: min_obstacle_margin(tr, spec.system)))(trajectories)
    )
    per_rollout_goal_distances = np.asarray(
        jax.jit(jax.vmap(lambda tr: final_goal_distances(tr, spec.system, spec.xbar)))(trajectories)
    )

    trajectory_finite = np.isfinite(trajectories_np).all(axis=tuple(range(1, trajectories_np.ndim)))
    control_finite = np.isfinite(controls_np).all(axis=tuple(range(1, controls_np.ndim)))
    cost_finite = np.isfinite(costs_np)
    finite_rollouts = trajectory_finite & control_finite & cost_finite
    finite_indices = np.flatnonzero(finite_rollouts)
    nonfinite_indices = np.flatnonzero(~finite_rollouts)
    per_step_trajectory_finite = np.isfinite(trajectories_np).all(axis=-1)
    first_nonfinite_step: list[int | None] = []
    for rollout_steps, is_finite in zip(per_step_trajectory_finite, trajectory_finite, strict=False):
        if is_finite:
            first_nonfinite_step.append(None)
        else:
            first_nonfinite_step.append(int(np.flatnonzero(~rollout_steps)[0]))

    def finite_stat(values: np.ndarray, reducer: str = "mean") -> float | None:
        finite_values = np.asarray(values)[np.isfinite(values)]
        if finite_values.size == 0:
            return None
        if reducer == "mean":
            return float(finite_values.mean())
        if reducer == "std":
            return float(finite_values.std())
        if reducer == "sum":
            return float(finite_values.sum())
        if reducer == "min":
            return float(finite_values.min())
        if reducer == "max":
            return float(finite_values.max())
        raise ValueError(f"Unknown finite_stat reducer={reducer!r}.")

    def rollout_stat(values: np.ndarray, reducer: str = "mean") -> float | None:
        values = np.asarray(values)
        if finite_indices.size == 0:
            return None
        return finite_stat(values[finite_indices], reducer)

    def rollout_axis_mean(values: np.ndarray) -> list[float | None]:
        values = np.asarray(values)
        if finite_indices.size == 0:
            return [None] * int(values.shape[-1])
        return [finite_stat(values[finite_indices, idx], "mean") for idx in range(values.shape[-1])]

    summary = {
        "system": spec.system.name,
        "n_rollouts": int(args.n_rollouts),
        "eval_batch_size": int(eval_batch_size),
        "finite_rollouts": int(finite_rollouts.sum()),
        "nonfinite_rollouts": int((~finite_rollouts).sum()),
        "nonfinite_rollout_indices": nonfinite_indices.tolist(),
        "first_nonfinite_trajectory_step": first_nonfinite_step,
        "t_end": int(spec.t_end),
        "state_dim": int(trajectories_np.shape[-1]),
        "control_dim": int(controls_np.shape[-1]),
        "mean_cost": rollout_stat(costs_np, "mean"),
        "std_cost": rollout_stat(costs_np, "std"),
        "total_collisions": rollout_stat(per_rollout_collisions, "sum"),
        "mean_collisions": rollout_stat(per_rollout_collisions, "mean"),
        "total_obstacle_violations": rollout_stat(per_rollout_obstacle_violations, "sum"),
        "mean_obstacle_violations": rollout_stat(per_rollout_obstacle_violations, "mean"),
        "min_obstacle_margin": rollout_stat(per_rollout_min_obstacle_margin, "min"),
        "mean_final_goal_distance": rollout_stat(per_rollout_goal_distances, "mean"),
        "max_final_goal_distance": rollout_stat(per_rollout_goal_distances, "max"),
        "per_agent_mean_final_goal_distance": rollout_axis_mean(per_rollout_goal_distances),
        "controller_input_clip": float(spec.controller_input_clip),
        "control_squash": bool(spec.control_squash),
        "control_min": finite_stat(controls_np, "min"),
        "control_max": finite_stat(controls_np, "max"),
        "alpha_terminal": float(spec.alpha_terminal),
        "initial_states_file": "initial_states.npy",
        "disturbances_file": "disturbances.npy",
        "qvel_impulse": (
            asdict(spec.system.qvel_impulse)
            if spec.system.qvel_impulse is not None else None
        ),
    }

    for metric in spec.task.metrics:
        metric_type = str(metric.type)
        metric_name = metric.name or metric_type
        if metric_type == "cost":
            summary[metric_name] = rollout_stat(costs_np, "mean")
        elif metric_type == "collisions":
            summary[metric_name] = rollout_stat(per_rollout_collisions, "mean")
        elif metric_type == "obstacle_violations":
            summary[metric_name] = rollout_stat(per_rollout_obstacle_violations, "mean")
        elif metric_type == "min_obstacle_margin":
            summary[metric_name] = rollout_stat(per_rollout_min_obstacle_margin, "min")
        elif metric_type == "final_goal_distance":
            summary[metric_name] = rollout_stat(per_rollout_goal_distances, "mean")
        else:
            summary[metric_name] = None

    with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    with open(output_dir / "system_config.json", "w", encoding="utf-8") as f:
        json.dump(spec.system.to_dict(), f, indent=2)

    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
