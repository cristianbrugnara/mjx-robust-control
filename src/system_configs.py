"""
Dataclasses describing a system (MJCF + JSON). 
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco

Array = jax.Array


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(k), _freeze_json(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_json(v) for v in value)
    return value


@dataclass(frozen=True)
class ObstacleSpec:
    """Smooth ellipsoid obstacle used by the differentiable task loss.

    The MJCF still defines the real dynamics/visuals. These obstacle specs are
    only the differentiable training semantics used by the objective.
    """

    center: tuple[float, ...]
    radii: tuple[float, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class QVelImpulseSpec:
    """Optional per-rollout velocity impulse applied during MJX rollout."""

    step: int
    indices: tuple[int, ...]
    sample_low: tuple[float, ...]
    sample_high: tuple[float, ...]
    apply_to_prediction: bool = False


@dataclass(frozen=True)
class ControlInterfaceSpec:
    """Policy-control semantics and policy-to-actuator transform parameters."""

    type: str = "direct_actuator"
    params: Any = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: Any | None) -> "ControlInterfaceSpec":
        """Parse an optional control-interface block."""
        if raw is None:
            return cls()
        if isinstance(raw, ControlInterfaceSpec):
            return raw
        return cls(
            type=str(raw.get("type", "direct_actuator")),
            params=_freeze_json(raw.get("params", {})),
        )


@dataclass(frozen=True)
class ReferenceSpec:
    """Named task reference, stored inline or loaded from a sidecar array file."""

    name: str
    value: Any | None = None
    path: str | None = None


@dataclass(frozen=True)
class CostTermSpec:
    """JSON-configurable differentiable cost primitive."""

    type: str
    weight: float = 1.0
    where: str = "running"
    params: Any = None


@dataclass(frozen=True)
class ControllerInputSpec:
    """One block in the REN input signal."""

    type: str
    scale: float = 1.0
    clip: float | None = None
    params: Any = None


@dataclass(frozen=True)
class MetricSpec:
    """Evaluation metric requested by a task config."""

    type: str
    name: str | None = None
    params: Any = None


@dataclass(frozen=True)
class TaskSpec:
    """JSON-first task definition consumed by rollout/train/evaluate."""

    cost_terms: tuple[CostTermSpec, ...] = ()
    controller_inputs: tuple[ControllerInputSpec, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    references: tuple[ReferenceSpec, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TaskSpec":
        """Parse a task block from JSON data."""
        if raw is None:
            return cls()
        raw = dict(raw)
        return cls(
            cost_terms=tuple(
                term if isinstance(term, CostTermSpec) else CostTermSpec(
                    type=str(term["type"]),
                    weight=float(term.get("weight", 1.0)),
                    where=str(term.get("where", "running")),
                    params=_freeze_json(term.get("params", {})),
                )
                for term in raw.get("cost_terms", ())
            ),
            controller_inputs=tuple(
                block if isinstance(block, ControllerInputSpec) else ControllerInputSpec(
                    type=str(block["type"]),
                    scale=float(block.get("scale", 1.0)),
                    clip=(
                        float(block["clip"])
                        if block.get("clip") is not None else None
                    ),
                    params=_freeze_json(block.get("params", {})),
                )
                for block in raw.get("controller_inputs", ())
            ),
            metrics=tuple(
                metric if isinstance(metric, MetricSpec) else MetricSpec(
                    type=str(metric["type"]),
                    name=(
                        str(metric["name"])
                        if metric.get("name") is not None else None
                    ),
                    params=_freeze_json(metric.get("params", {})),
                )
                for metric in raw.get("metrics", ())
            ),
            references=tuple(
                ref if isinstance(ref, ReferenceSpec) else ReferenceSpec(
                    name=str(ref["name"]),
                    value=ref.get("value"),
                    path=(
                        str(ref["path"])
                        if ref.get("path") is not None else None
                    ),
                )
                for ref in raw.get("references", ())
            ),
        )

    def validate(self, system: "SystemSpec") -> None:
        """Validate task references, costs, metrics, and input blocks."""
        seen_refs: set[str] = set()
        for ref in self.references:
            if not ref.name:
                raise ValueError(f"System '{system.name}' has a task reference with an empty name.")
            if ref.name in seen_refs:
                raise ValueError(f"System '{system.name}' has duplicate task reference '{ref.name}'.")
            seen_refs.add(ref.name)
            if (ref.value is None) == (ref.path is None):
                raise ValueError(
                    f"System '{system.name}' task reference '{ref.name}' must set exactly one of value/path."
                )

        for term in self.cost_terms:
            if term.type not in _VALID_COST_TERMS:
                raise ValueError(f"System '{system.name}' has unknown task cost type '{term.type}'.")
            if term.where not in _VALID_COST_WHERE:
                raise ValueError(
                    f"System '{system.name}' task cost '{term.type}' has invalid where='{term.where}'."
                )
            params = dict(term.params or ())
            for site_name in params.get("sites", ()):
                if not str(site_name):
                    raise ValueError(f"System '{system.name}' task cost '{term.type}' has an empty site name.")

        for block in self.controller_inputs:
            if block.type not in _VALID_INPUT_BLOCKS:
                raise ValueError(
                    f"System '{system.name}' has unknown controller input block '{block.type}'."
                )
            if block.scale < 0.0:
                raise ValueError(
                    f"System '{system.name}' controller input '{block.type}' has negative scale."
                )
            params = dict(block.params or ())
            for site_name in params.get("sites", ()):
                if not str(site_name):
                    raise ValueError(f"System '{system.name}' controller input '{block.type}' has an empty site name.")


_VALID_COST_TERMS = {
    "state_l2",
    "state_bounds",
    "control_l2",
    "pairwise_distance_barrier",
    "ellipsoid_obstacle",
    "box_bounds",
    "road_network",
    "heading_to_goal",
    "planar_heading_velocity",
}


_VALID_COST_WHERE = {"running", "terminal", "both"}
_VALID_INPUT_BLOCKS = {
    "state",
    "state_error",
    "imc_residual",
}


@dataclass(frozen=True)
class SystemSpec:
    """Everything task-specific that the rollout/training code needs.

    For every agent, the flat state is

        [qpos entries..., qvel entries...]

    with the configured number of qpos and qvel entries per agent.

    Public systems define their controller inputs, costs, metrics, and
    references explicitly in JSON through ``task``.
    """

    name: str
    description: str

    n_agents: int
    dof_per_entity: int
    controls_per_entity: int
    position_indices: tuple[int, ...]
    state_labels_per_entity: tuple[str, ...]
    control_labels_per_entity: tuple[str, ...]

    x0: tuple[float, ...]
    xbar: tuple[float, ...]
    q_diag_per_entity: tuple[float, ...]
    init_noise_mask_per_entity: tuple[float, ...]

    t_end: int
    learning_rate: float
    epochs: int
    n_xi: int
    l: int
    n_traj: int
    std_ini: float

    min_dist: float
    agent_radius: float
    collision_security_margin: float
    bounds: tuple[float, ...] | None
    obstacles: tuple[ObstacleSpec, ...]
    obstacle_threshold_per_agent: float

    pre_stab_K: float
    pre_stab_mode: str
    pre_stab_control_indices: tuple[int, ...]

    output_amplification: float
    psi_u_inner_output_gain: float
    std_ini_param: float
    use_sp: bool = False

    qpos_idx: tuple[int, ...] | None = None
    qvel_idx: tuple[int, ...] | None = None
    task: TaskSpec = field(default_factory=TaskSpec)
    qpos_dim_per_entity: int | None = None
    qvel_dim_per_entity: int | None = None
    quaternion_indices_per_entity: tuple[tuple[int, int, int, int], ...] = ()
    control_center: tuple[float, ...] = ()
    policy_control_low: tuple[float, ...] = ()
    policy_control_high: tuple[float, ...] = ()
    control_interface: ControlInterfaceSpec = field(default_factory=ControlInterfaceSpec)
    qvel_impulse: QVelImpulseSpec | None = None
    mjx_disable_constraints: bool = False

    @property
    def qpos_dim_per_entity_resolved(self) -> int:
        return self.dof_per_entity if self.qpos_dim_per_entity is None else int(self.qpos_dim_per_entity)

    @property
    def qvel_dim_per_entity_resolved(self) -> int:
        return self.dof_per_entity if self.qvel_dim_per_entity is None else int(self.qvel_dim_per_entity)

    @property
    def entity_state_dim(self) -> int:
        return self.qpos_dim_per_entity_resolved + self.qvel_dim_per_entity_resolved

    @property
    def state_dim(self) -> int:
        return self.n_agents * self.entity_state_dim

    @property
    def control_dim(self) -> int:
        return self.n_agents * self.controls_per_entity

    def x0_array(self, *, dtype=jnp.float32) -> Array:
        """Return x0 as a JAX array."""
        return jnp.asarray(self.x0, dtype=dtype)

    def xbar_array(self, *, dtype=jnp.float32) -> Array:
        """Return xbar as a JAX array."""
        return jnp.asarray(self.xbar, dtype=dtype)

    def init_noise_mask(self, *, dtype=jnp.float32) -> Array:
        """Return the tiled initial-condition noise mask."""
        return jnp.tile(jnp.asarray(self.init_noise_mask_per_entity, dtype=dtype), self.n_agents)

    def Q(self, *, dtype=jnp.float32) -> Array:
        """Return the block-diagonal state cost matrix."""
        per_entity = jnp.diag(jnp.asarray(self.q_diag_per_entity, dtype=dtype))
        return jnp.kron(jnp.eye(self.n_agents, dtype=dtype), per_entity)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the system spec as plain dataclass data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SystemSpec":
        """Parse a system spec from JSON data."""
        raw = dict(raw)
        raw["position_indices"] = tuple(raw["position_indices"])
        raw["state_labels_per_entity"] = tuple(raw["state_labels_per_entity"])
        raw["control_labels_per_entity"] = tuple(raw["control_labels_per_entity"])
        raw["x0"] = tuple(float(v) for v in raw["x0"])
        raw["xbar"] = tuple(float(v) for v in raw["xbar"])
        raw["q_diag_per_entity"] = tuple(float(v) for v in raw["q_diag_per_entity"])
        raw["init_noise_mask_per_entity"] = tuple(
            float(v) for v in raw["init_noise_mask_per_entity"]
        )
        if raw.get("bounds") is not None:
            raw["bounds"] = tuple(float(v) for v in raw["bounds"])
        raw["obstacles"] = tuple(
            obs if isinstance(obs, ObstacleSpec) else ObstacleSpec(
                center=tuple(float(v) for v in obs["center"]),
                radii=tuple(float(v) for v in obs["radii"]),
                weight=float(obs.get("weight", 1.0)),
            )
            for obs in raw.get("obstacles", ())
        )
        for _removed in (
            "alpha_x", "alpha_u", "alpha_ca", "alpha_obst", "alpha_side", "alpha_barrier",
            "alpha_site", "alpha_site_terminal", "alpha_site_floor", "alpha_site_floor_terminal",
            "site_targets", "site_floor_sites", "site_floor_min_z", "runtime_site_trajectory",
        ):
            raw.pop(_removed, None)
        raw["pre_stab_control_indices"] = tuple(raw.get("pre_stab_control_indices", ()))
        if raw.get("qpos_idx") is not None:
            raw["qpos_idx"] = tuple(int(v) for v in raw["qpos_idx"])
        if raw.get("qvel_idx") is not None:
            raw["qvel_idx"] = tuple(int(v) for v in raw["qvel_idx"])
        raw["task"] = TaskSpec.from_dict(raw.get("task"))
        if raw.get("qpos_dim_per_entity") is not None:
            raw["qpos_dim_per_entity"] = int(raw["qpos_dim_per_entity"])
        if raw.get("qvel_dim_per_entity") is not None:
            raw["qvel_dim_per_entity"] = int(raw["qvel_dim_per_entity"])
        raw["quaternion_indices_per_entity"] = tuple(
            tuple(int(v) for v in block)
            for block in raw.get("quaternion_indices_per_entity", ())
        )
        raw["control_center"] = tuple(float(v) for v in raw.get("control_center", ()))
        raw["policy_control_low"] = tuple(float(v) for v in raw.get("policy_control_low", ()))
        raw["policy_control_high"] = tuple(float(v) for v in raw.get("policy_control_high", ()))
        raw["control_interface"] = ControlInterfaceSpec.from_dict(raw.get("control_interface"))
        raw["mjx_disable_constraints"] = bool(raw.get("mjx_disable_constraints", False))
        if raw.get("qvel_impulse") is not None:
            impulse = raw["qvel_impulse"]
            raw["qvel_impulse"] = (
                impulse if isinstance(impulse, QVelImpulseSpec)
                else QVelImpulseSpec(
                    step=int(impulse["step"]),
                    indices=tuple(int(v) for v in impulse["indices"]),
                    sample_low=tuple(float(v) for v in impulse["sample_low"]),
                    sample_high=tuple(float(v) for v in impulse["sample_high"]),
                    apply_to_prediction=bool(impulse.get("apply_to_prediction", False)),
                )
            )
        return cls(**raw)

    def validate_basic(self) -> None:
        """Validate JSON-level dimensions and parameter ranges."""
        if len(self.state_labels_per_entity) != self.entity_state_dim:
            raise ValueError(
                f"System '{self.name}' has {len(self.state_labels_per_entity)} state labels per "
                f"entity, expected {self.entity_state_dim}."
            )
        if len(self.control_labels_per_entity) != self.controls_per_entity:
            raise ValueError(
                f"System '{self.name}' has {len(self.control_labels_per_entity)} control labels per "
                f"entity, expected {self.controls_per_entity}."
            )
        if self.control_center and len(self.control_center) != self.control_dim:
            raise ValueError(
                f"System '{self.name}' control_center must have length {self.control_dim}; "
                f"got {len(self.control_center)}."
            )
        if bool(self.policy_control_low) != bool(self.policy_control_high):
            raise ValueError(
                f"System '{self.name}' must set both policy_control_low and policy_control_high, or neither."
            )
        if self.policy_control_low and len(self.policy_control_low) != self.control_dim:
            raise ValueError(
                f"System '{self.name}' policy_control_low must have length {self.control_dim}; "
                f"got {len(self.policy_control_low)}."
            )
        if self.policy_control_high and len(self.policy_control_high) != self.control_dim:
            raise ValueError(
                f"System '{self.name}' policy_control_high must have length {self.control_dim}; "
                f"got {len(self.policy_control_high)}."
            )
        if any(lo > hi for lo, hi in zip(self.policy_control_low, self.policy_control_high)):
            raise ValueError(f"System '{self.name}' has policy_control_low > policy_control_high.")
        if len(self.x0) != self.state_dim or len(self.xbar) != self.state_dim:
            raise ValueError(
                f"System '{self.name}' x0/xbar lengths must be {self.state_dim}; got "
                f"{len(self.x0)} and {len(self.xbar)}."
            )
        if len(self.q_diag_per_entity) != self.entity_state_dim:
            raise ValueError(
                f"System '{self.name}' q_diag_per_entity must have length {self.entity_state_dim}."
            )
        if len(self.init_noise_mask_per_entity) != self.entity_state_dim:
            raise ValueError(
                f"System '{self.name}' init_noise_mask_per_entity must have length "
                f"{self.entity_state_dim}."
            )
        if any(i < 0 or i >= self.entity_state_dim for i in self.position_indices):
            raise ValueError(f"System '{self.name}' has invalid position_indices.")
        if self.bounds is not None and len(self.bounds) != 2 * len(self.position_indices):
            raise ValueError(
                f"System '{self.name}' bounds must have {2 * len(self.position_indices)} values."
            )
        for obs in self.obstacles:
            if len(obs.center) != len(self.position_indices) or len(obs.radii) != len(self.position_indices):
                raise ValueError(
                    f"System '{self.name}' obstacle dimensions must match position_indices."
                )
            if any(r <= 0.0 for r in obs.radii):
                raise ValueError(f"System '{self.name}' obstacle radii must be positive.")
        for block in self.quaternion_indices_per_entity:
            if len(block) != 4:
                raise ValueError(
                    f"System '{self.name}' quaternion index blocks must have length 4."
                )
            if any(i < 0 or i >= self.entity_state_dim for i in block):
                raise ValueError(f"System '{self.name}' has invalid quaternion indices.")
        if self.qpos_dim_per_entity_resolved <= 0 or self.qvel_dim_per_entity_resolved <= 0:
            raise ValueError(f"System '{self.name}' qpos/qvel dimensions must be positive.")
        if self.qvel_impulse is not None:
            impulse = self.qvel_impulse
            if impulse.step < 0 or impulse.step >= self.t_end:
                raise ValueError(
                    f"System '{self.name}' qvel_impulse.step must be in [0, {self.t_end})."
                )
            if not impulse.indices:
                raise ValueError(f"System '{self.name}' qvel_impulse.indices cannot be empty.")
            if len(impulse.indices) != len(impulse.sample_low) or len(impulse.indices) != len(impulse.sample_high):
                raise ValueError(
                    f"System '{self.name}' qvel_impulse indices/sample_low/sample_high lengths must match."
                )
            if any(i < 0 or i >= self.n_agents * self.qvel_dim_per_entity_resolved for i in impulse.indices):
                raise ValueError(f"System '{self.name}' qvel_impulse has an invalid qvel index.")
            if any(lo > hi for lo, hi in zip(impulse.sample_low, impulse.sample_high)):
                raise ValueError(f"System '{self.name}' qvel_impulse has sample_low > sample_high.")
        self.task.validate(self)
        if self.pre_stab_mode not in ("none", "direct_position", "quadrotor_position"):
            raise ValueError(
                f"System '{self.name}' pre_stab_mode must be 'none', 'direct_position', "
                "or 'quadrotor_position'."
            )
        if self.pre_stab_mode == "direct_position":
            if len(self.pre_stab_control_indices) != len(self.position_indices):
                raise ValueError(
                    f"System '{self.name}' direct_position pre-stabilizer needs one control "
                    "index per position coordinate."
                )
            if any(i < 0 or i >= self.controls_per_entity for i in self.pre_stab_control_indices):
                raise ValueError(f"System '{self.name}' has invalid pre_stab_control_indices.")

    def resolve_qpos_idx(self, nq: int) -> tuple[int, ...]:
        """Return the qpos indices used by the flat controller state."""
        expected = self.n_agents * self.qpos_dim_per_entity_resolved
        idx = tuple(range(nq)) if self.qpos_idx is None else tuple(self.qpos_idx)
        if len(idx) != expected:
            raise ValueError(
                f"System '{self.name}' needs {expected} qpos entries, but qpos_idx has "
                f"length {len(idx)}. If the MJCF has extra joints, set qpos_idx in the system config."
            )
        return idx

    def resolve_qvel_idx(self, nv: int) -> tuple[int, ...]:
        """Return the qvel indices used by the flat controller state."""
        expected = self.n_agents * self.qvel_dim_per_entity_resolved
        idx = tuple(range(nv)) if self.qvel_idx is None else tuple(self.qvel_idx)
        if len(idx) != expected:
            raise ValueError(
                f"System '{self.name}' needs {expected} qvel entries, but qvel_idx has "
                f"length {len(idx)}. If the MJCF has extra joints, set qvel_idx in the system config."
            )
        return idx

    def validate_against_mj_model(self, mj_model: Any) -> None:
        """Check the system spec against a loaded MuJoCo model."""
        qpos_idx = self.resolve_qpos_idx(int(mj_model.nq))
        qvel_idx = self.resolve_qvel_idx(int(mj_model.nv))
        if max(qpos_idx, default=-1) >= int(mj_model.nq):
            raise ValueError(f"System '{self.name}' qpos_idx exceeds mj_model.nq={mj_model.nq}.")
        if max(qvel_idx, default=-1) >= int(mj_model.nv):
            raise ValueError(f"System '{self.name}' qvel_idx exceeds mj_model.nv={mj_model.nv}.")
        if self.control_interface.type == "direct_actuator" and int(mj_model.nu) != self.control_dim:
            raise ValueError(
                f"System '{self.name}' expects control_dim={self.control_dim}, but MJCF has "
                f"nu={mj_model.nu}. Keep actuator order one block per agent, or update the config."
            )
        if self.control_interface.type in ("bicycle_steering", "quadrotor_attitude_mixer", "quadrotor_wrench_mixer"):
            params = dict(self.control_interface.params)
            default_actuators_per_entity = 3 if self.control_interface.type == "bicycle_steering" else self.controls_per_entity
            actuators_per_entity = int(params.get("actuators_per_entity", default_actuators_per_entity))
            expected_nu = self.n_agents * actuators_per_entity
            if int(mj_model.nu) != expected_nu:
                raise ValueError(
                    f"System '{self.name}' control_interface={self.control_interface.type!r} "
                    f"currently expects MJCF nu={expected_nu}, got {mj_model.nu}."
                )
        if self.control_interface.type not in (
            "direct_actuator",
            "bicycle_steering",
            "quadrotor_attitude_mixer",
            "quadrotor_wrench_mixer",
        ):
            raise ValueError(
                f"System '{self.name}' has unknown control_interface type "
                f"{self.control_interface.type!r}."
            )
        if self.control_center and self.policy_control_low and self.policy_control_high:
            for i, value in enumerate(self.control_center):
                low, high = self.policy_control_low[i], self.policy_control_high[i]
                if value < low or value > high:
                    raise ValueError(
                        f"System '{self.name}' control_center[{i}]={value} is outside "
                        f"policy range [{low}, {high}]."
                    )
        elif self.control_center and self.control_interface.type == "direct_actuator":
            center = tuple(float(v) for v in self.control_center)
            limited = getattr(mj_model, "actuator_ctrllimited", ())
            ctrlrange = getattr(mj_model, "actuator_ctrlrange", ())
            for i, value in enumerate(center):
                if bool(limited[i]):
                    low, high = float(ctrlrange[i][0]), float(ctrlrange[i][1])
                    if value < low or value > high:
                        raise ValueError(
                            f"System '{self.name}' control_center[{i}]={value} is outside "
                            f"MJCF ctrlrange [{low}, {high}]."
                        )
def apply_mjx_model_options(spec: SystemSpec, mj_model: Any) -> None:
    """Apply in-memory MJX compatibility options without editing the MJCF file."""
    if spec.mjx_disable_constraints:
        mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)


def load_system_spec(name_or_path: str | None = None) -> SystemSpec:
    """Load a system JSON by path or by name from assets/config."""
    if name_or_path is None:
        name_or_path = "corridor"

    candidate = Path(name_or_path)
    if not candidate.exists() and candidate.suffix.lower() != ".json":
        candidate = Path("assets") / "config" / f"{name_or_path}.json"

    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            spec = SystemSpec.from_dict(json.load(f))
        spec.validate_basic()
        return spec

    raise ValueError(
        f"Unknown system '{name_or_path}'. Pass a JSON path or add assets/config/{name_or_path}.json."
    )


def save_system_spec(spec: SystemSpec, path: str | Path) -> None:
    """Write a system spec to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2)


__all__ = [
    "ControlInterfaceSpec",
    "ControllerInputSpec",
    "CostTermSpec",
    "MetricSpec",
    "ObstacleSpec",
    "ReferenceSpec",
    "SystemSpec",
    "TaskSpec",
    "apply_mjx_model_options",
    "load_system_spec",
    "save_system_spec",
]
