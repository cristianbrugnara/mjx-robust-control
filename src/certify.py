"""
Probabilistic certification. Evaluates a trained controller on a held-out
certification set, sorts rollout costs, computes the DKW confidence radius epsilon_m and
the order statistic index k_star, and produces a distribution-free cost threshold with
confidence 1-delta.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import equinox as eqx
import jax
import mujoco
import numpy as np
from mujoco import mjx

from evaluate import (
        build_controller_skeleton,
        build_rollout_config,
        load_eval_spec,
    )
from workflow_utils import (
        build_data_init,
        sample_initial_conditions,
        sample_qvel_impulses,
    )
from jax_rollout import rollout_with_trajectory
from system_configs import apply_mjx_model_options


DEFAULT_OBJECTIVES = ("mean", "cvar", "pinball", "softmax", "worst_case")


@dataclass(frozen=True)
class CertificationResult:
    threshold: float
    epsilon_m: float
    k_star: int
    m: int
    alpha: float
    delta: float


def epsilon_m(m: int, delta: float) -> float:
    """DKW confidence radius for m samples."""
    if m <= 0:
        raise ValueError("m must be positive.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * float(m)))


def k_star(m: int, alpha: float, eps: float) -> int:
    """Order statistic index used by the certificate."""
    if m <= 0:
        raise ValueError("m must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    return int(math.ceil(float(m) * (1.0 - float(alpha) + float(eps))))


def theorem1_threshold(values: Iterable[float], *, alpha: float, delta: float) -> CertificationResult:
    """Certify a cost threshold from calibration samples."""
    vals = sorted(float(v) for v in values)
    m = len(vals)
    eps = epsilon_m(m, delta)
    ks = k_star(m, alpha, eps)
    if ks > m or alpha < eps:
        raise ValueError(
            "sample-size condition failed: "
            f"m={m}, k_star={ks}, alpha={alpha:.6g}, epsilon_m={eps:.6g}. "
        )
    return CertificationResult(
        threshold=float(vals[ks - 1]),
        epsilon_m=float(eps),
        k_star=int(ks),
        m=int(m),
        alpha=float(alpha),
        delta=float(delta),
    )


def split_pass_fail(costs: Iterable[float], threshold: float) -> tuple[list[int], list[int]]:
    respect: list[int] = []
    violate: list[int] = []
    for idx, cost in enumerate(costs):
        if float(cost) <= float(threshold):
            respect.append(idx)
        else:
            violate.append(idx)
    return respect, violate


def select_examples(
    indices: Iterable[int],
    *,
    costs: Iterable[float],
    threshold: float,
    n_examples: int,
    prefer: str,
) -> list[int]:
    idx = [int(i) for i in indices]
    if n_examples <= 0 or not idx:
        return []
    cost_list = [float(v) for v in costs]
    if prefer == "respect":
        idx.sort(key=lambda i: abs(cost_list[i] - threshold))
    elif prefer == "violate":
        idx.sort(key=lambda i: (cost_list[i] - threshold, cost_list[i]), reverse=True)
    else:
        raise ValueError("prefer must be 'respect' or 'violate'.")
    return idx[: int(n_examples)]


def json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def np_save(path: Path, value: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value))


def load_numpy(path: Path) -> Any:
    import numpy as np

    return np.load(path)


def evaluate_checkpoint(
    *,
    xml_path: Path,
    checkpoint_path: Path,
    sys_model: str | None,
    system_config_path: str | None,
    seed: int,
    n_rollouts: int,
) -> dict[str, Any]:

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    spec = load_eval_spec(
        checkpoint_path=str(checkpoint_path),
        sys_model=sys_model,
        system_config_path=str(system_config_path) if system_config_path is not None else None,
        mj_model=mj_model,
    )
    apply_mjx_model_options(spec.system, mj_model)
    mjx_model = mjx.put_model(mj_model)
    data_template = mjx.make_data(mjx_model)
    rollout_config = build_rollout_config(spec)
    controller_skeleton = build_controller_skeleton(spec, mj_model)
    controller = eqx.tree_deserialise_leaves(str(checkpoint_path), controller_skeleton)

    key = jax.random.PRNGKey(int(seed))
    key_x0, key_impulse = jax.random.split(key, 2)
    x0_batch = sample_initial_conditions(
        key_x0,
        spec.x0,
        std_ini=float(spec.std_ini),
        n_samples=int(n_rollouts),
        init_noise_mask=spec.init_noise_mask,
        n_agents=spec.system.n_agents,
        entity_state_dim=spec.system.entity_state_dim,
        quaternion_indices_per_entity=spec.system.quaternion_indices_per_entity,
    )
    qvel_impulse_batch = sample_qvel_impulses(
        key_impulse,
        spec.system,
        n_samples=int(n_rollouts),
        dtype=spec.x0.dtype,
    )

    x_prediction0 = spec.xbar
    xi_dim = controller.psi_u.n_xi
    ctrl_dim = int(mj_model.nu)

    def single_eval(x_real0: Any, qvel_impulse: Any) -> tuple[Any, Any, Any]:
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

    costs, trajectories, controls = jax.jit(jax.vmap(single_eval))(
        x0_batch,
        qvel_impulse_batch,
    )
    return {
        "spec": spec,
        "costs": np.asarray(costs),
        "trajectories": np.asarray(trajectories),
        "controls": np.asarray(controls),
        "initial_states": np.asarray(x0_batch),
        "disturbances": np.asarray(qvel_impulse_batch),
    }


def train_checkpoint(
    *,
    objective: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    """Train a checkpoint for one robust objective."""
    from train import TrainConfig, train as run_train

    config = TrainConfig(
        xml_path=str(args.xml_path),
        sys_model=args.sys_model,
        system_config_path=str(args.system_config_path) if args.system_config_path is not None else None,
        seed=int(args.seed),
        n_train=int(args.n_train),
        n_valid=int(args.n_valid),
        validation_period=int(args.validation_period),
        save_path=str(checkpoint_path),
        epochs=int(args.epochs) if args.epochs is not None else None,
        learning_rate=args.learning_rate,
        batch_size=int(args.batch_size),
        objective=objective,
        cvar_alpha=float(args.train_alpha),
        pinball_quantile=float(1.0 - args.train_alpha),
        softmax_beta=float(args.softmax_beta),
        controller_input_clip=float(args.controller_input_clip),
        control_squash=args.control_squash,
        control_margin=float(args.control_margin),
        grad_clip_norm=args.grad_clip_norm,
        resample_train_batch=bool(args.resample_train_batch),
    )
    run_train(config)


def maybe_train_checkpoint(
    *,
    objective: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    """Train only when the expected checkpoint and metadata are missing."""
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    if checkpoint_path.exists() and meta_path.exists() and not args.force_train:
        print(f"[{objective}] reusing checkpoint {checkpoint_path}")
        return
    print(f"[{objective}] training checkpoint {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_checkpoint(objective=objective, checkpoint_path=checkpoint_path, args=args)


def save_rollout_bundle(
    *,
    out_dir: Path,
    prefix: str,
    result: dict[str, Any],
) -> None:
    np_save(out_dir / f"{prefix}_costs.npy", result["costs"])
    np_save(out_dir / f"{prefix}_trajectories.npy", result["trajectories"])
    np_save(out_dir / f"{prefix}_controls.npy", result["controls"])
    np_save(out_dir / f"{prefix}_initial_states.npy", result["initial_states"])
    np_save(out_dir / f"{prefix}_disturbances.npy", result["disturbances"])


def save_selected_trajectory_file(
    *,
    out_dir: Path,
    name: str,
    eval_result: dict[str, Any],
    indices: list[int],
) -> None:

    trajectories = np.asarray(eval_result["trajectories"])
    controls = np.asarray(eval_result["controls"])
    costs = np.asarray(eval_result["costs"])
    disturbances = np.asarray(eval_result["disturbances"])
    idx = np.asarray(indices, dtype=np.int64)

    np_save(out_dir / f"{name}_indices.npy", idx)
    np_save(out_dir / f"{name}_trajectories.npy", trajectories[idx] if idx.size else trajectories[:0])
    np_save(out_dir / f"{name}_controls.npy", controls[idx] if idx.size else controls[:0])
    np_save(out_dir / f"{name}_costs.npy", costs[idx] if idx.size else costs[:0])
    np_save(out_dir / f"{name}_disturbances.npy", disturbances[idx] if idx.size else disturbances[:0])


def plot_thresholds(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    names = [item["objective"] for item in summaries]
    thresholds = [float(item["threshold"]) for item in summaries]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(names)), 4.0))
    ax.bar(range(len(names)), thresholds)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("certified threshold")
    ax.set_title("Theorem 1 thresholds")
    fig.tight_layout()
    fig.savefig(run_dir / "thresholds.png", dpi=160)
    plt.close(fig)


def empirical_cdf(values: Any, grid: Any) -> Any:
    vals = np.asarray(values).reshape(1, -1)
    t_grid = np.asarray(grid).reshape(-1, 1)
    return (vals <= t_grid).mean(axis=1)


def plot_cdfs(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    all_costs = []
    for item in summaries:
        all_costs.append(load_numpy(Path(item["cert_costs_path"])))
    merged = np.concatenate([np.asarray(v).reshape(-1) for v in all_costs])
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    pad = 0.05 * (hi - lo + 1.0e-12)
    grid = np.linspace(lo - pad, hi + pad, 500)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for item, costs in zip(summaries, all_costs):
        fhat = empirical_cdf(costs, grid)
        line = ax.plot(grid, fhat, label=item["objective"])[0]
        eps = float(item["epsilon_m"])
        ax.fill_between(
            grid,
            np.clip(fhat - eps, 0.0, 1.0),
            np.clip(fhat + eps, 0.0, 1.0),
            alpha=0.10,
            color=line.get_color(),
        )
        ax.axvline(float(item["threshold"]), color=line.get_color(), linestyle="--", alpha=0.45)
    ax.axhline(1.0 - float(summaries[0]["alpha"]), color="black", linestyle=":", alpha=0.8)
    ax.set_xlabel("threshold t")
    ax.set_ylabel("empirical CDF on cert set")
    ax.set_title("Certification CDFs with DKW bands")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "cert_cdfs.png", dpi=160)
    plt.close(fig)

    for item, costs in zip(summaries, all_costs):
        fhat = empirical_cdf(costs, grid)
        eps = float(item["epsilon_m"])
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(grid, fhat, label="empirical CDF")
        ax.fill_between(grid, np.clip(fhat - eps, 0.0, 1.0), np.clip(fhat + eps, 0.0, 1.0), alpha=0.18)
        ax.axvline(float(item["threshold"]), color="black", linestyle="--", label="cert threshold")
        ax.axhline(1.0 - float(item["alpha"]), color="black", linestyle=":", label="1-alpha")
        ax.set_xlabel("threshold t")
        ax.set_ylabel("empirical CDF on cert set")
        ax.set_title(f"{item['objective']} certification CDF")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / item["objective"] / "cert_cdf.png", dpi=160)
        plt.close(fig)


def copy_checkpoint_metadata(checkpoint_path: Path, out_dir: Path) -> None:
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    if checkpoint_path.exists():
        shutil.copy2(checkpoint_path, out_dir / checkpoint_path.name)
    if meta_path.exists():
        shutil.copy2(meta_path, out_dir / meta_path.name)


def run_objective(objective: str, *, run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Train if needed, evaluate certification samples, and save one objective summary."""
    out_dir = run_dir / objective
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_dir) / f"{args.sys_model}_{objective}_seed{args.seed}.eqx"

    maybe_train_checkpoint(objective=objective, checkpoint_path=checkpoint_path, args=args)
    copy_checkpoint_metadata(checkpoint_path, out_dir)

    cert_seed = int(args.seed) + int(args.cert_seed_offset)
    eval_seed = int(args.seed) + int(args.eval_seed_offset)
    print(f"[{objective}] evaluating cert rollouts seed={cert_seed}, n={args.m_cert}")
    cert_result = evaluate_checkpoint(
        xml_path=Path(args.xml_path),
        checkpoint_path=checkpoint_path,
        sys_model=args.sys_model,
        system_config_path=Path(args.system_config_path) if args.system_config_path is not None else None,
        seed=cert_seed,
        n_rollouts=int(args.m_cert),
    )
    print(f"[{objective}] evaluating held-out rollouts seed={eval_seed}, n={args.n_eval}")
    eval_result = evaluate_checkpoint(
        xml_path=Path(args.xml_path),
        checkpoint_path=checkpoint_path,
        sys_model=args.sys_model,
        system_config_path=Path(args.system_config_path) if args.system_config_path is not None else None,
        seed=eval_seed,
        n_rollouts=int(args.n_eval),
    )

    save_rollout_bundle(out_dir=out_dir, prefix="cert", result=cert_result)
    save_rollout_bundle(out_dir=out_dir, prefix="eval", result=eval_result)

    cert = theorem1_threshold(cert_result["costs"], alpha=float(args.alpha), delta=float(args.delta))
    respect, violate = split_pass_fail(eval_result["costs"], cert.threshold)
    selected_respect = select_examples(
        respect,
        costs=eval_result["costs"],
        threshold=cert.threshold,
        n_examples=int(args.n_examples),
        prefer="respect",
    )
    selected_violate = select_examples(
        violate,
        costs=eval_result["costs"],
        threshold=cert.threshold,
        n_examples=int(args.n_examples),
        prefer="violate",
    )

    np_save(out_dir / "respect_indices.npy", respect)
    np_save(out_dir / "violate_indices.npy", violate)
    np_save(out_dir / "selected_respect_indices.npy", selected_respect)
    np_save(out_dir / "selected_violate_indices.npy", selected_violate)
    save_selected_trajectory_file(out_dir=out_dir, name="selected_respect", eval_result=eval_result, indices=selected_respect)
    save_selected_trajectory_file(out_dir=out_dir, name="selected_violate", eval_result=eval_result, indices=selected_violate)

    eval_costs = np.asarray(eval_result["costs"])
    viz_layout_args = (
        f"--qpos_dim_per_entity {cert_result['spec'].system.qpos_dim_per_entity_resolved} "
        f"--qvel_dim_per_entity {cert_result['spec'].system.qvel_dim_per_entity_resolved}"
    )
    summary = {
        "objective": objective,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_copy": str(out_dir / checkpoint_path.name),
        "alpha": float(args.alpha),
        "delta": float(args.delta),
        "m_cert": int(args.m_cert),
        "n_eval": int(args.n_eval),
        "epsilon_m": float(cert.epsilon_m),
        "k_star": int(cert.k_star),
        "threshold": float(cert.threshold),
        "eval_respect_count": int(len(respect)),
        "eval_violate_count": int(len(violate)),
        "eval_respect_fraction": float(len(respect) / max(1, int(args.n_eval))),
        "eval_mean_cost": float(np.mean(eval_costs)),
        "eval_max_cost": float(np.max(eval_costs)),
        "selected_respect_indices": selected_respect,
        "selected_violate_indices": selected_violate,
        "cert_costs_path": str(out_dir / "cert_costs.npy"),
        "eval_costs_path": str(out_dir / "eval_costs.npy"),
        "cert_trajectories_path": str(out_dir / "cert_trajectories.npy"),
        "eval_trajectories_path": str(out_dir / "eval_trajectories.npy"),
        "cert_disturbances_path": str(out_dir / "cert_disturbances.npy"),
        "eval_disturbances_path": str(out_dir / "eval_disturbances.npy"),
        "selected_respect_trajectories_path": str(out_dir / "selected_respect_trajectories.npy"),
        "selected_violate_trajectories_path": str(out_dir / "selected_violate_trajectories.npy"),
        "visualize_respect_command": (
            f"uv run python src/visualize.py --xml_path {args.xml_path} "
            f"--trajectories_path {out_dir / 'selected_respect_trajectories.npy'} "
            f"{viz_layout_args} --selection sequential --top_down --show_traces --playback_speed 0.35"
        ),
        "visualize_violate_command": (
            f"uv run python src/visualize.py --xml_path {args.xml_path} "
            f"--trajectories_path {out_dir / 'selected_violate_trajectories.npy'} "
            f"{viz_layout_args} --selection sequential --top_down --show_traces --playback_speed 0.35"
        ),
    }
    json_dump(out_dir / "certification_summary.json", summary)
    print(
        f"[{objective}] threshold={cert.threshold:.6g} "
        f"eval_respect_fraction={summary['eval_respect_fraction']:.3f} "
        f"violations={len(violate)}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, and certify JAX/MJX REN controllers.")
    parser.add_argument("--xml_path", type=Path, default=Path("assets/mjcf/corridor.xml"))
    parser.add_argument("--sys_model", type=str, default="corridor")
    parser.add_argument("--system_config_path", type=Path, default=Path("assets/config/corridor.json"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--objectives", nargs="+", choices=DEFAULT_OBJECTIVES, default=list(DEFAULT_OBJECTIVES))
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/certification"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("artifacts/certification_checkpoints"))
    parser.add_argument("--force_train", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--validation_period", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--resample_train_batch", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--train_alpha", type=float, default=None)
    parser.add_argument("--m_cert", type=int, default=300)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--cert_seed_offset", type=int, default=100_000)
    parser.add_argument("--eval_seed_offset", type=int, default=200_000)
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--softmax_beta", type=float, default=10.0)

    parser.add_argument("--controller_input_clip", type=float, default=1.0)
    parser.set_defaults(control_squash=None)
    parser.add_argument("--control_squash", dest="control_squash", action="store_true")
    parser.add_argument("--no_control_squash", dest="control_squash", action="store_false")
    parser.add_argument("--control_margin", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    args = parser.parse_args()
    if args.train_alpha is None:
        args.train_alpha = args.alpha
    return args


def main() -> None:
    """Run certification for the requested objectives and save plots and summaries."""
    args = parse_args()
    theorem1_threshold([0.0] * int(args.m_cert), alpha=float(args.alpha), delta=float(args.delta))

    run_name = args.run_name or f"{args.sys_model}_seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    json_dump(
        run_dir / "certification_config.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "differentiability_note": (
                "Training losses are built from JAX/MJX rollouts: mjx.step is inside "
                "jax.lax.scan and gradients flow through the simulated dynamics."
            ),
        },
    )

    summaries = [run_objective(objective, run_dir=run_dir, args=args) for objective in args.objectives]
    json_dump(run_dir / "summary.json", {"run_dir": str(run_dir), "objectives": summaries})
    plot_thresholds(run_dir, summaries)
    plot_cdfs(run_dir, summaries)

    print(f"Saved certification run to {run_dir}")
    print("Example replay commands:")
    for item in summaries:
        print(item["visualize_respect_command"])
        if item["eval_violate_count"] > 0:
            print(item["visualize_violate_command"])


if __name__ == "__main__":
    main()
