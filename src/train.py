"""
Training entry point. Loads an XML/JSON system, builds a REN controller, vmaps differentiable
MJX rollouts over a batch, and optimises with Optax under a chosen robust objective.
Saves the best-validation checkpoint as an Equinox .eqx file with a .meta.json file for metadata.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mujoco
import numpy as np
import optax
from mujoco import mjx
from tqdm.auto import tqdm

from jax_models import Controller
from jax_rollout import RolloutConfig, rollout
from robust_objectives import (
    cvar_loss,
    mean_loss,
    objective_requires_tau,
    pinball_loss,
    softmax_loss,
    worst_case_loss,
)
from system_configs import (
    apply_mjx_model_options,
    SystemSpec,
    load_system_spec,
)
from workflow_utils import (
    actuator_ctrl_bounds,
    build_data_init,
    build_rollout_config,
    controller_input_dim_from_blocks,
    policy_ctrl_bounds,
    require_explicit_task,
    resolve_task_references,
    sample_initial_conditions,
    sample_qvel_impulses,
)

Array = jax.Array


@dataclass(frozen=True)
class TrainConfig:
    xml_path: str = "corridor.xml"
    sys_model: str = "corridor"
    system_config_path: str | None = None
    seed: int = 3
    n_train: int = 100
    n_valid: int = 100
    validation_period: int = 50
    save_path: str | None = None

    epochs: int | None = None
    learning_rate: float | None = None
    std_ini_override: float | None = None
    n_xi_override: int | None = None
    l_override: int | None = None
    batch_size: int = 100

    std_ini_param: float | None = None
    use_sp: bool | None = None
    pre_stab_K: float | None = None
    dof_per_entity: int | None = None

    objective: str = "mean"
    cvar_alpha: float = 0.10
    pinball_quantile: float = 0.90
    softmax_beta: float = 10.0

    output_amplification: float | None = None
    psi_u_inner_output_gain: float | None = None

    controller_input_clip: float = 1.0
    control_squash: bool | None = None
    control_margin: float = 0.05
    alpha_terminal: float | None = None
    grad_clip_norm: float | None = 1.0

    resample_train_batch: bool = False
    verbose: bool = False


def zero_nominal_prediction(t: int | Array, y: Array, u: Array) -> Array:
    del t, u
    return y


def load_training_system(config: TrainConfig) -> SystemSpec:
    return load_system_spec(config.system_config_path or config.sys_model)


def build_objective_fn(config: TrainConfig) -> Callable[[Array, Array], Array]:
    if config.objective == "mean":
        return lambda costs, tau: mean_loss(costs)
    if config.objective == "cvar":
        return lambda costs, tau: cvar_loss(costs, config.cvar_alpha, tau)
    if config.objective == "pinball":
        return lambda costs, tau: pinball_loss(costs, config.pinball_quantile, tau)
    if config.objective == "softmax":
        return lambda costs, tau: softmax_loss(costs, config.softmax_beta)
    if config.objective == "worst_case":
        return lambda costs, tau: worst_case_loss(costs)
    raise ValueError(f"Unknown objective '{config.objective}'.")


def build_loss_fn(
    static_controller: Controller,
    *,
    mjx_model: Any,
    data_template: Any,
    rollout_config: RolloutConfig,
    t_end: int,
    ctrl_dim: int,
    x_prediction0: Array,
    objective_fn: Callable[[Array, Array], Array],
) -> Callable[[dict[str, Array], tuple[Array, Array] | Array], Array]:
    xi_dim = static_controller.psi_u.n_xi

    def single_rollout_loss(
        controller_params: Any,
        x_real0: Array,
        qvel_impulse: Array,
    ) -> Array:
        controller = eqx.combine(controller_params, static_controller)
        data_init = build_data_init(
            data_template,
            x_real0,
            x_prediction0,
            xi_dim=xi_dim,
            ctrl_dim=ctrl_dim,
            policy_ctrl_dim=int(rollout_config.ctrl_low.shape[0]),
            dof_per_entity=rollout_config.dof_per_entity,
            qpos_dim_per_entity=rollout_config.qpos_dim_per_entity,
            qvel_dim_per_entity=rollout_config.qvel_dim_per_entity,
        )
        return rollout(
            controller,
            mjx_model,
            data_init,
            t_end,
            rollout_config,
            qvel_impulse,
        )

    vmapped_rollout = jax.vmap(
        lambda x_real0, qvel_impulse, controller_params: single_rollout_loss(
            controller_params, x_real0, qvel_impulse
        ),
        in_axes=(0, 0, None),
    )

    def loss_fn(
        trainable: dict[str, Array],
        batch: tuple[Array, Array],
    ) -> Array:
        batch_x0, batch_qvel_impulses = batch
        losses = vmapped_rollout(
            batch_x0,
            batch_qvel_impulses,
            trainable["controller"],
        )
        return objective_fn(losses, trainable["tau"])

    return loss_fn


def make_train_step(
    loss_fn: Callable[[Any, Any], Array],
    optimizer: optax.GradientTransformation,
) -> Callable[[Any, Any, Any], tuple[Any, Any, Array]]:
    """JIT one optimizer step and skip updates with non-finite gradients."""
    def tree_all_finite(tree) -> Array:
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return jnp.asarray(True)
        finite = jnp.asarray(True)
        for leaf in leaves:
            if leaf is not None:
                finite = finite & jnp.all(jnp.isfinite(leaf))
        return finite

    @jax.jit
    def train_step(trainable, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(trainable, batch)
        finite = jnp.isfinite(loss) & tree_all_finite(grads)

        def apply_update(args):
            trainable_in, opt_state_in = args
            updates, opt_state_out = optimizer.update(grads, opt_state_in, trainable_in)
            return optax.apply_updates(trainable_in, updates), opt_state_out

        trainable, opt_state = jax.lax.cond(
            finite,
            apply_update,
            lambda args: args,
            (trainable, opt_state),
        )
        return trainable, opt_state, loss

    return train_step


def make_eval_step(loss_fn: Callable[[Any, Any], Array]) -> Callable[[Any, Any], Array]:
    @jax.jit
    def eval_step(trainable, batch):
        return loss_fn(trainable, batch)

    return eval_step


def build_default_save_path(config: TrainConfig, spec: SystemSpec | None = None) -> Path:
    system_name = spec.name if spec is not None else config.sys_model
    return Path("trained_models") / f"{system_name}_{config.objective}_seed{config.seed}.eqx"


def _jsonify(value: Any) -> Any:
    """Convert arrays, dataclasses, and paths into JSON-friendly values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    try:
        arr = jnp.asarray(value)
        return arr.tolist()
    except Exception:
        return str(value)


def save_metadata(save_path: Path, metadata: dict[str, Any]) -> None:
    meta_path = Path(str(save_path) + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(metadata), f, indent=2)


def resolve_training_values(config: TrainConfig, spec: SystemSpec) -> dict[str, Any]:
    """Merge CLI overrides with JSON defaults for training and rollout knobs."""
    return {
        "learning_rate": spec.learning_rate if config.learning_rate is None else config.learning_rate,
        "epochs": spec.epochs if config.epochs is None else config.epochs,
        "std_ini": spec.std_ini if config.std_ini_override is None else config.std_ini_override,
        "n_xi": spec.n_xi if config.n_xi_override is None else config.n_xi_override,
        "l": spec.l if config.l_override is None else config.l_override,
        "std_ini_param": spec.std_ini_param if config.std_ini_param is None else config.std_ini_param,
        "pre_stab_K": spec.pre_stab_K if config.pre_stab_K is None else config.pre_stab_K,
        "use_sp": spec.use_sp if config.use_sp is None else config.use_sp,
        "output_amplification": (
            spec.output_amplification if config.output_amplification is None else config.output_amplification
        ),
        "psi_u_inner_output_gain": (
            spec.psi_u_inner_output_gain
            if config.psi_u_inner_output_gain is None
            else config.psi_u_inner_output_gain
        ),
        "controller_input_clip": config.controller_input_clip,
        "control_squash": True if config.control_squash is None else bool(config.control_squash),
        "control_margin": config.control_margin,
        "alpha_terminal": 5.0 if config.alpha_terminal is None else config.alpha_terminal,
        "grad_clip_norm": config.grad_clip_norm,
    }


def make_optimizer(learning_rate: float, grad_clip_norm: float | None) -> optax.GradientTransformation:
    if grad_clip_norm is None or grad_clip_norm <= 0.0:
        return optax.adam(learning_rate)
    return optax.chain(optax.clip_by_global_norm(float(grad_clip_norm)), optax.adam(learning_rate))


def train(config: TrainConfig) -> tuple[Controller, float, float]:
    """Train one controller and save the best validation checkpoint."""
    spec = load_training_system(config)
    if config.dof_per_entity is not None and config.dof_per_entity != spec.dof_per_entity:
        raise ValueError(
            f"--dof_per_entity={config.dof_per_entity} conflicts with system '{spec.name}' "
            f"dof_per_entity={spec.dof_per_entity}. Update the system config instead."
        )

    resolved = resolve_training_values(config, spec)
    resolved = {**resolved, "alpha_terminal": 0.0 if config.alpha_terminal is None else config.alpha_terminal}
    task = require_explicit_task(spec)

    x0 = spec.x0_array()
    xbar = spec.xbar_array(dtype=x0.dtype)
    Q = spec.Q(dtype=x0.dtype)

    save_path = Path(config.save_path) if config.save_path is not None else build_default_save_path(config, spec)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mj_model = mujoco.MjModel.from_xml_path(config.xml_path)
    spec.validate_against_mj_model(mj_model)
    apply_mjx_model_options(spec, mj_model)
    mjx_model = mjx.put_model(mj_model)
    data_template = mjx.make_data(mjx_model)
    actuator_ctrl_low, actuator_ctrl_high = actuator_ctrl_bounds(mj_model, dtype=x0.dtype)
    ctrl_low, ctrl_high = policy_ctrl_bounds(
        spec,
        actuator_ctrl_low,
        actuator_ctrl_high,
        dtype=x0.dtype,
    )

    qpos_idx = jnp.asarray(spec.resolve_qpos_idx(mj_model.nq), dtype=jnp.int32)
    qvel_idx = jnp.asarray(spec.resolve_qvel_idx(mj_model.nv), dtype=jnp.int32)
    task_base_dir = (
        Path(config.system_config_path).resolve().parent
        if config.system_config_path is not None
        else Path.cwd()
    )
    task_references = resolve_task_references(task, base_dir=task_base_dir, dtype=x0.dtype)
    rollout_config = build_rollout_config(
        system=spec,
        xbar=xbar,
        Q=Q,
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        pre_stab_K=float(resolved["pre_stab_K"]),
        alpha_terminal=float(resolved["alpha_terminal"]),
        controller_input_clip=float(resolved["controller_input_clip"]),
        control_squash=bool(resolved["control_squash"]),
        control_margin=float(resolved["control_margin"]),
        ctrl_low=ctrl_low,
        ctrl_high=ctrl_high,
        actuator_ctrl_low=actuator_ctrl_low,
        actuator_ctrl_high=actuator_ctrl_high,
        task=task,
        task_references=task_references,
    )

    key = jr.PRNGKey(config.seed)
    (
        key_model,
        key_train,
        key_valid,
        key_batch,
        key_train_impulse,
        key_valid_impulse,
        key_online_x0,
        key_online_impulse,
    ) = jr.split(key, 8)

    init_mask = spec.init_noise_mask(dtype=x0.dtype)
    train_x0 = sample_initial_conditions(
        key_train,
        x0,
        std_ini=float(resolved["std_ini"]),
        n_samples=config.n_train,
        init_noise_mask=init_mask,
        n_agents=spec.n_agents,
        entity_state_dim=spec.entity_state_dim,
        quaternion_indices_per_entity=spec.quaternion_indices_per_entity,
    )
    valid_x0 = sample_initial_conditions(
        key_valid,
        x0,
        std_ini=float(resolved["std_ini"]),
        n_samples=config.n_valid,
        init_noise_mask=init_mask,
        n_agents=spec.n_agents,
        entity_state_dim=spec.entity_state_dim,
        quaternion_indices_per_entity=spec.quaternion_indices_per_entity,
    )
    train_qvel_impulses = sample_qvel_impulses(
        key_train_impulse,
        spec,
        n_samples=config.n_train,
        dtype=x0.dtype,
    )
    valid_qvel_impulses = sample_qvel_impulses(
        key_valid_impulse,
        spec,
        n_samples=config.n_valid,
        dtype=x0.dtype,
    )
    input_dim = controller_input_dim_from_blocks(
        state_dim=int(xbar.shape[0]),
        controller_inputs=task.controller_inputs,
    )

    controller = Controller(
        zero_nominal_prediction,
        n=input_dim,
        m=int(spec.control_dim),
        n_xi=int(resolved["n_xi"]),
        l=int(resolved["l"]),
        key=key_model,
        use_sp=bool(resolved["use_sp"]),
        t_end_sp=int(spec.t_end),
        std_ini_param=float(resolved["std_ini_param"]),
        output_amplification=float(resolved["output_amplification"]),
        psi_u_inner_output_gain=float(resolved["psi_u_inner_output_gain"]),
    )

    controller_params, static_controller = eqx.partition(controller, eqx.is_inexact_array)
    tau0 = jnp.asarray(0.0, dtype=x0.dtype)
    trainable = {
        "controller": controller_params,
        "tau": tau0,
    }

    optimizer = make_optimizer(float(resolved["learning_rate"]), resolved.get("grad_clip_norm"))
    opt_state = optimizer.init(trainable)
    objective_fn = build_objective_fn(config)

    loss_fn = build_loss_fn(
        static_controller,
        mjx_model=mjx_model,
        data_template=data_template,
        rollout_config=rollout_config,
        t_end=int(spec.t_end),
        ctrl_dim=int(mj_model.nu),
        x_prediction0=xbar,
        objective_fn=objective_fn,
    )
    train_step = make_train_step(loss_fn, optimizer)
    eval_step = make_eval_step(loss_fn)

    best_val = jnp.inf
    best_trainable = {
        "controller": jax.tree_util.tree_map(lambda x: x.copy(), trainable["controller"]),
        "tau": trainable["tau"].copy(),
    }
    saved_best = False

    batch_size = int(max(1, min(config.batch_size, train_x0.shape[0])))

    pbar = tqdm(range(int(resolved["epochs"])), desc="Training", unit="epoch")

    for epoch in pbar:
        if config.resample_train_batch:
            batch_x0 = sample_initial_conditions(
                jr.fold_in(key_online_x0, epoch),
                x0,
                std_ini=float(resolved["std_ini"]),
                n_samples=batch_size,
                init_noise_mask=init_mask,
                n_agents=spec.n_agents,
                entity_state_dim=spec.entity_state_dim,
                quaternion_indices_per_entity=spec.quaternion_indices_per_entity,
            )
            batch_qvel_impulses = sample_qvel_impulses(
                jr.fold_in(key_online_impulse, epoch),
                spec,
                n_samples=batch_size,
                dtype=x0.dtype,
            )
        elif batch_size == train_x0.shape[0]:
            batch_x0 = train_x0
            batch_qvel_impulses = train_qvel_impulses
        else:
            epoch_key = jr.fold_in(key_batch, epoch)
            inds = jr.permutation(epoch_key, train_x0.shape[0])[:batch_size]
            batch_x0 = train_x0[inds]
            batch_qvel_impulses = train_qvel_impulses[inds]

        train_batch = (
            batch_x0,
            batch_qvel_impulses,
        )
        trainable, opt_state, train_loss = train_step(trainable, opt_state, train_batch)

        should_validate = (epoch % config.validation_period == 0) or (
            epoch == int(resolved["epochs"]) - 1
        )

        if should_validate:
            val_loss = eval_step(
                trainable,
                (
                    valid_x0,
                    valid_qvel_impulses,
                ),
            )

            if float(val_loss) < float(best_val):
                best_val = val_loss
                best_trainable = {
                    "controller": jax.tree_util.tree_map(lambda x: x.copy(), trainable["controller"]),
                    "tau": trainable["tau"].copy(),
                }
                best_controller = eqx.combine(best_trainable["controller"], static_controller)
                eqx.tree_serialise_leaves(save_path, best_controller)
                saved_best = True
                save_metadata(
                    save_path,
                    {
                        "train_config": asdict(config),
                        "system_config": spec.to_dict(),
                        "resolved": {
                            **{k: _jsonify(v) for k, v in resolved.items()},
                            "best_tau": float(best_trainable["tau"]),
                            "batch_size": int(batch_size),
                        },
                        "controller": {
                            "n": int(input_dim),
                            "m": int(spec.control_dim),
                            "n_xi": int(resolved["n_xi"]),
                            "l": int(resolved["l"]),
                            "use_sp": bool(resolved["use_sp"]),
                            "std_ini_param": float(resolved["std_ini_param"]),
                            "t_end_sp": int(spec.t_end),
                            "output_amplification": float(resolved["output_amplification"]),
                            "psi_u_inner_output_gain": float(resolved["psi_u_inner_output_gain"]),
                        },
                        "rollout": {
                            "task": task,
                            "task_references": [(name, value) for name, value in task_references],
                            "controller_input_clip": float(resolved["controller_input_clip"]),
                            "control_squash": bool(resolved["control_squash"]),
                            "control_margin": float(resolved["control_margin"]),
                            "ctrl_low": ctrl_low,
                            "ctrl_high": ctrl_high,
                            "actuator_ctrl_low": actuator_ctrl_low,
                            "actuator_ctrl_high": actuator_ctrl_high,
                            "control_center": rollout_config.control_center,
                            "control_interface": spec.control_interface,
                            "qvel_impulse": spec.qvel_impulse,
                            "alpha_terminal": float(resolved["alpha_terminal"]),
                            "controller_input_dim": int(input_dim),
                            "grad_clip_norm": resolved["grad_clip_norm"],
                        },
                        "task": {
                            "x0": x0,
                            "xbar": xbar,
                            "Q": Q,
                            "t_end": int(spec.t_end),
                            "n_agents": int(spec.n_agents),
                            "std_ini": float(resolved["std_ini"]),
                            "init_noise_mask": init_mask,
                            "dof_per_entity": int(spec.dof_per_entity),
                            "qpos_dim_per_entity": int(spec.qpos_dim_per_entity_resolved),
                            "qvel_dim_per_entity": int(spec.qvel_dim_per_entity_resolved),
                            "qpos_idx": qpos_idx,
                            "qvel_idx": qvel_idx,
                            "pre_stab_K": float(resolved["pre_stab_K"]),
                            "alpha_terminal": float(resolved["alpha_terminal"]),
                        },
                    },
                )

            pbar.set_postfix(
                train_loss=f"{float(train_loss):.6f}",
                val_loss=f"{float(val_loss):.6f}",
                best_val=f"{float(best_val):.6f}",
                tau=f"{float(trainable['tau']):.6f}"
                if objective_requires_tau(config.objective)
                    else "",
            )

            if config.verbose:
                tau_msg = f" tau={float(trainable['tau']):.6f}" if objective_requires_tau(config.objective) else ""
                print(
                    f"epoch={epoch:05d}/{int(resolved['epochs']) - 1} "
                    f"system={spec.name} "
                    f"objective={config.objective} "
                    f"u_squash={bool(resolved['control_squash'])} "
                    f"alpha_terminal={float(resolved['alpha_terminal']):.3f} "
                    f"train_loss={float(train_loss):.6f} "
                    f"val_loss={float(val_loss):.6f} "
                    f"best_val={float(best_val):.6f}"
                    f"{tau_msg}"
                )

    if not saved_best or not np.isfinite(float(best_val)) or not save_path.exists():
        raise RuntimeError(
            "Training did not produce a finite validation checkpoint. "
            f"No usable model was saved at {save_path}. "
            "Try a smaller learning rate/std_ini, mean objective first, or inspect rollout stability."
        )

    best_controller = eqx.combine(best_trainable["controller"], static_controller)
    return best_controller, float(best_val), float(best_trainable["tau"])


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="corridor.xml")
    parser.add_argument("--sys_model", type=str, default="corridor")
    parser.add_argument("--system_config_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument("--validation_period", type=int, default=50)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--std_ini_override", type=float, default=None)
    parser.add_argument("--n_xi_override", type=int, default=None)
    parser.add_argument("--l_override", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--std_ini_param", type=float, default=None)
    parser.add_argument("--pre_stab_K", type=float, default=None)
    parser.add_argument("--use_sp", action="store_true", default=None)
    parser.add_argument("--dof_per_entity", type=int, default=None)
    parser.add_argument("--output_amplification", type=float, default=None)
    parser.add_argument("--psi_u_inner_output_gain", type=float, default=None)
    parser.add_argument("--controller_input_clip", type=float, default=1.0)
    parser.set_defaults(control_squash=None)
    parser.add_argument("--control_squash", dest="control_squash", action="store_true")
    parser.add_argument("--no_control_squash", dest="control_squash", action="store_false")
    parser.add_argument("--control_margin", type=float, default=0.05)
    parser.add_argument("--alpha_terminal", type=float, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--resample_train_batch",
        action="store_true",
        help="Sample a fresh randomized train batch every epoch instead of reusing the finite train set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one detailed line every validation_period. Tqdm remains enabled either way.",
    )

    parser.add_argument(
        "--objective",
        type=str,
        choices=["mean", "cvar", "pinball", "softmax", "worst_case"],
        default="mean",
    )
    parser.add_argument("--cvar_alpha", type=float, default=0.10)
    parser.add_argument("--pinball_quantile", type=float, default=0.90)
    parser.add_argument("--softmax_beta", type=float, default=10.0)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    spec = load_training_system(config)
    _, best_val, best_tau = train(config)
    resolved_path = config.save_path if config.save_path is not None else str(build_default_save_path(config, spec))
    print(f"saved_best_model={resolved_path}")
    if objective_requires_tau(config.objective):
        print(f"best_tau={best_tau:.6f}")
    print(f"best_validation_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
