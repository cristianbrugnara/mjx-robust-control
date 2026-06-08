"""
Trajectory replay tool. Loads .npy trajectory files produced by evaluate.py and replays
them in the MuJoCo GUI. Supports sequential or random rollout selection, side-by-side
model comparison, trace overlays, top-down view, and optional video export.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import mujoco
import mujoco.viewer
import numpy as np

from jax_rollout import state_to_qpos_qvel


@dataclass(frozen=True)
class TrajectorySet:
    label: str
    path: Path
    trajectories: np.ndarray
    costs: np.ndarray | None

    @property
    def n_rollouts(self) -> int:
        return int(self.trajectories.shape[0])

    @property
    def t_end(self) -> int:
        return int(self.trajectories.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.trajectories.shape[2])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for trajectory replay."""
    parser = argparse.ArgumentParser(
        description="Replay saved evaluate.py trajectories in MuJoCo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Sequential loop through one model:\n"
            "    uv run python src/visualize.py --xml_path assets/mjcf/corridor.xml \\\n"
            "      --trajectories_path eval_results/model_a/trajectories.npy \\\n"
            "      --selection sequential --loop\n\n"
            "  Random rollout each loop:\n"
            "    uv run python src/visualize.py --xml_path assets/mjcf/corridor.xml \\\n"
            "      --trajectories_path eval_results/model_a/trajectories.npy \\\n"
            "      --selection random --loop --seed 0\n\n"
            "  Side-by-side comparison, same rollout index in each result directory:\n"
            "    uv run python src/visualize.py --xml_path assets/mjcf/corridor.xml \\\n"
            "      --trajectories_path eval_results/model_a/trajectories.npy \\\n"
            "      --compare_trajectories_path eval_results/model_b/trajectories.npy \\\n"
            "      --selection sequential --loop\n\n"
            "  Three planar drones, with traces:\n"
            "    uv run python src/visualize.py --xml_path assets/mjcf/drones3_3d.xml \\\n"
            "      --trajectories_path eval_results/model_a/trajectories.npy \\\n"
            "      --dof_per_entity 3 --show_traces --top_down\n"
        ),
    )

    parser.add_argument("--xml_path", type=str, required=True, help="Path to the MJCF file.")
    parser.add_argument(
        "--trajectories_path",
        type=str,
        required=True,
        help="Path to trajectories.npy saved by evaluate.py for the primary model.",
    )
    parser.add_argument(
        "--compare_trajectories_path",
        type=str,
        default=None,
        help=(
            "Optional second trajectories.npy. When provided, both models are shown "
            "side by side using the same rollout index."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Console label for the primary trajectory set. Defaults to its parent directory name.",
    )
    parser.add_argument(
        "--compare_label",
        type=str,
        default=None,
        help="Console label for the comparison trajectory set. Defaults to its parent directory name.",
    )
    parser.add_argument(
        "--rollout_idx",
        type=int,
        default=0,
        help="Starting rollout index. In --selection single mode, this is the only rollout shown.",
    )
    parser.add_argument(
        "--selection",
        choices=("single", "sequential", "random"),
        default="sequential",
        help=(
            "Which rollout to show after each playback. "
            "single repeats one rollout; sequential advances by one; "
            "random samples a new rollout. Default: sequential."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate MJCF, trajectory layout, comparison compatibility, and trace colors without opening a viewer.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=None,
        help=(
            "Optional maximum number of rollout clips to play before exiting. "
            "Useful with sequential/random when --loop is not set."
        ),
    )
    parser.add_argument(
        "--playback_speed",
        type=float,
        default=1.0,
        help="Playback speed as a multiple of real time. Must be > 0.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Keep playing clips. With --selection sequential this cycles through rollouts; "
            "with --selection random this samples another rollout each time."
        ),
    )
    parser.add_argument(
        "--pause_between_rollouts",
        type=float,
        default=0.25,
        help="Seconds to pause after finishing each rollout before loading the next one.",
    )
    parser.add_argument(
        "--side_by_side_spacing",
        type=float,
        default=8.0,
        help=(
            "Horizontal distance between the primary and comparison copies when "
            "--compare_trajectories_path is used."
        ),
    )
    parser.add_argument(
        "--top_down",
        action="store_true",
        help="Start the viewer with a top-down camera.",
    )
    parser.add_argument(
        "--dof_per_entity",
        type=int,
        default=2,
        help=(
            "Shorthand for both qpos and qvel dimensions per entity. "
            "Use 3 for the planar drone XML where each drone has [x, y, theta]."
        ),
    )
    parser.add_argument(
        "--qpos_dim_per_entity",
        type=int,
        default=None,
        help="MuJoCo qpos entries per entity. Overrides --dof_per_entity when set.",
    )
    parser.add_argument(
        "--qvel_dim_per_entity",
        type=int,
        default=None,
        help="MuJoCo qvel entries per entity. Overrides --dof_per_entity when set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed used by --selection random.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-rollout console output.")

    parser.add_argument("--show_traces", action="store_true", help="Draw rollout traces.")
    parser.add_argument("--trace_width", type=float, default=0.025, help="Trace capsule radius.")
    parser.add_argument("--trace_stride", type=int, default=2, help="Draw every Nth trace point.")
    parser.add_argument("--trace_alpha", type=float, default=0.72, help="Trace opacity in [0, 1].")
    parser.add_argument("--trace_z", type=float, default=0.08, help="Z height for trace drawing.")
    parser.add_argument(
        "--print_trace_colors",
        action="store_true",
        help="Print inferred per-agent trace colors for debugging.",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        help=(
            "Directory for offscreen video export. When a comparison trajectory is provided, "
            "one separate MP4 is written for each trajectory set using the same camera path."
        ),
    )
    parser.add_argument(
        "--record_path",
        type=str,
        default=None,
        help=(
            "Output MP4 path or filename prefix for offscreen export. With comparison or "
            "--record_split_rollouts, label/rollout suffixes are added automatically."
        ),
    )
    parser.add_argument(
        "--record_split_rollouts",
        action="store_true",
        help="Write one video per selected rollout instead of concatenating rollouts per dataset.",
    )
    parser.add_argument(
        "--record_fps",
        type=float,
        default=None,
        help=(
            "Video frame rate. Defaults to playback_speed / model timestep, so the saved "
            "video duration matches the interactive replay speed."
        ),
    )
    parser.add_argument("--record_width", type=int, default=1920, help="Export video width in pixels.")
    parser.add_argument("--record_height", type=int, default=1080, help="Export video height in pixels.")
    parser.add_argument(
        "--record_camera",
        choices=("orbit", "angled", "side", "top_down", "fixed"),
        default="orbit",
        help=(
            "Camera used for offscreen videos. 'orbit' is a smooth moving free camera; "
            "'fixed' uses --record_camera_name from the MJCF."
        ),
    )
    parser.add_argument(
        "--record_camera_name",
        type=str,
        default="angled",
        help="MJCF fixed-camera name used when --record_camera fixed.",
    )
    parser.add_argument(
        "--record_orbit_start",
        type=float,
        default=35.0,
        help="Starting azimuth in degrees for --record_camera orbit.",
    )
    parser.add_argument(
        "--record_orbit_degrees",
        type=float,
        default=180.0,
        help="Total azimuth sweep in degrees for --record_camera orbit.",
    )
    parser.add_argument(
        "--record_elevation",
        type=float,
        default=-28.0,
        help="Base camera elevation in degrees for offscreen free-camera modes.",
    )
    parser.add_argument(
        "--record_elevation_wobble",
        type=float,
        default=8.0,
        help="Smooth elevation variation in degrees for --record_camera orbit.",
    )
    parser.add_argument(
        "--record_distance",
        type=float,
        default=None,
        help="Free-camera distance. Defaults to a value derived from model.stat.extent.",
    )
    parser.add_argument(
        "--record_maxgeom",
        type=int,
        default=20000,
        help="Maximum number of geoms allocated for offscreen rendering, including traces.",
    )
    parser.add_argument(
        "--record_no_titles",
        action="store_true",
        help="Disable title overlays in exported videos.",
    )
    parser.add_argument(
        "--record_title_height",
        type=int,
        default=58,
        help="Height in pixels of the title band drawn at the top of exported videos.",
    )

    return parser.parse_args()


def default_label(path: Path) -> str:
    """Choose a readable label from a trajectory path."""
    if path.name == "trajectories.npy" and path.parent.name:
        return path.parent.name
    return path.stem


def load_trajectory_set(path_like: str, label: str | None = None) -> TrajectorySet:
    """Load trajectories and optional sidecar arrays from an eval directory."""
    path = Path(path_like)
    trajectories = np.load(path)
    if trajectories.ndim != 3:
        raise ValueError(
            f"Expected {path} to have shape (n_rollouts, T, n_state), got {trajectories.shape}."
        )

    costs_path = path.with_name("costs.npy")
    costs = np.load(costs_path) if costs_path.exists() else None
    if costs is not None:
        if costs.ndim != 1:
            raise ValueError(f"Expected {costs_path} with shape (n_rollouts,), got {costs.shape}.")
        if costs.shape[0] != trajectories.shape[0]:
            raise ValueError(
                f"{costs_path} has {costs.shape[0]} costs, but {path} has "
                f"{trajectories.shape[0]} rollouts."
            )

    return TrajectorySet(
        label=label or default_label(path),
        path=path,
        trajectories=np.asarray(trajectories),
        costs=np.asarray(costs) if costs is not None else None,
    )


def validate_compatible_sets(primary: TrajectorySet, compare: TrajectorySet | None) -> int:
    """Validate two trajectory sets can be replayed together."""
    if compare is None:
        return primary.n_rollouts
    if primary.state_dim != compare.state_dim:
        raise ValueError(
            f"State dimensions differ: {primary.label} has {primary.state_dim}, "
            f"{compare.label} has {compare.state_dim}."
        )
    return min(primary.n_rollouts, compare.n_rollouts)


def rollout_indices(
    *,
    n_rollouts: int,
    start_idx: int,
    selection: str,
    loop: bool,
    num_rollouts: int | None,
    seed: int,
) -> Iterator[int]:
    """Yield rollout indices according to the requested playback mode."""
    if n_rollouts <= 0:
        raise ValueError("No rollouts are available.")
    if start_idx < 0 or start_idx >= n_rollouts:
        raise IndexError(f"rollout_idx={start_idx} is out of bounds for {n_rollouts} rollouts.")
    if num_rollouts is not None and num_rollouts <= 0:
        raise ValueError("--num_rollouts must be positive when provided.")

    yielded = 0
    rng = np.random.default_rng(seed)
    idx = int(start_idx)

    while True:
        if selection == "single":
            next_idx = int(start_idx)
        elif selection == "sequential":
            next_idx = idx
            idx = (idx + 1) % n_rollouts
        elif selection == "random":
            next_idx = int(rng.integers(0, n_rollouts))
        else:
            raise ValueError(f"Unknown selection mode {selection!r}.")

        yield next_idx
        yielded += 1

        if num_rollouts is not None and yielded >= num_rollouts:
            return
        if not loop:
            if selection == "single":
                return
            if selection == "random" and num_rollouts is None:
                return
            if selection == "sequential" and num_rollouts is None and yielded >= n_rollouts:
                return


KINEMATIC_REF_ATTRS = {
    "body",
    "body1",
    "body2",
    "childbody",
    "geom",
    "geom1",
    "geom2",
    "joint",
    "joint1",
    "joint2",
    "site",
    "site1",
    "site2",
    "tendon",
    "actuator",
    "cranksite",
    "slidersite",
}


def prefix_local_names(elem: ET.Element, prefix: str) -> None:
    """Prefix MJCF names and local references inside one XML subtree."""
    if "name" in elem.attrib:
        elem.set("name", prefix + elem.attrib["name"])
    for attr in KINEMATIC_REF_ATTRS:
        if attr in elem.attrib:
            elem.set(attr, prefix + elem.attrib[attr])
    for child in list(elem):
        prefix_local_names(child, prefix)


def remove_if_present(root: ET.Element, tag: str) -> None:
    """Remove a top-level XML element if present."""
    node = root.find(tag)
    if node is not None:
        root.remove(node)


def xml_subtree_has_freejoint(elem: ET.Element) -> bool:
    """Return whether an MJCF subtree contains a free joint."""
    for node in elem.iter():
        if node.tag == "freejoint":
            return True
        if node.tag == "joint" and node.attrib.get("type") == "free":
            return True
    return False


def xml_has_freejoint(root: ET.Element) -> bool:
    """Return whether an MJCF root contains any free joint."""
    return xml_subtree_has_freejoint(root)


def offset_xml_pos(elem: ET.Element, offset_x: float) -> None:
    """Shift an MJCF element's pos attribute along world x."""
    xyz = [0.0, 0.0, 0.0]
    if "pos" in elem.attrib:
        parts = [float(v) for v in elem.attrib["pos"].split()]
        if len(parts) != 3:
            return
        xyz = parts
    xyz[0] += float(offset_x)
    elem.set("pos", " ".join(f"{v:.9g}" for v in xyz))


def build_recording_xml(xml_path: Path, *, offwidth: int, offheight: int) -> Path:
    """Create a temporary MJCF with a large enough offscreen framebuffer.

    MuJoCo's offscreen renderer checks ``<visual><global offwidth=... offheight=.../>``
    at model compile time.  The source MJCF can stay unchanged; this temporary
    copy only exists for video export.
    """
    offwidth = int(offwidth)
    offheight = int(offheight)
    if offwidth <= 0 or offheight <= 0:
        raise ValueError("offscreen framebuffer dimensions must be positive.")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(0, visual)

    global_node = visual.find("global")
    if global_node is None:
        global_node = ET.Element("global")
        visual.insert(0, global_node)

    current_width = int(float(global_node.attrib.get("offwidth", "0")))
    current_height = int(float(global_node.attrib.get("offheight", "0")))
    global_node.set("offwidth", str(max(current_width, offwidth)))
    global_node.set("offheight", str(max(current_height, offheight)))

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix=".visualize_recording_",
        dir=str(xml_path.parent),
        delete=False,
        encoding="utf-8",
    )
    with handle:
        handle.write(ET.tostring(root, encoding="unicode"))

    return Path(handle.name)


def build_side_by_side_xml(xml_path: Path, *, spacing: float, n_copies: int = 2) -> Path:
    """Create a temporary MJCF with repeated copies for comparison playback.

    MJCF free joints are only legal on top-level world bodies.  For models
    containing free joints we therefore cannot place each full copy inside a
    translated wrapper body.  Instead, free-joint bodies remain top-level,
    static world objects are shifted in XML, and replay offsets the x entries
    of qpos for each copied trajectory.
    """
    if n_copies < 2:
        raise ValueError("n_copies must be >= 2.")
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0 for side-by-side comparison.")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.set("model", f"{root.get('model', 'model')}_side_by_side")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{xml_path} does not contain a <worldbody> element.")

    original_world_children = [copy.deepcopy(child) for child in list(worldbody)]
    has_freejoint = any(xml_subtree_has_freejoint(child) for child in original_world_children)
    for child in list(worldbody):
        worldbody.remove(child)

    offsets = np.linspace(
        -0.5 * spacing * (n_copies - 1),
        0.5 * spacing * (n_copies - 1),
        n_copies,
    )

    for copy_idx, offset_x in enumerate(offsets):
        prefix = f"view{copy_idx}_"

        if not has_freejoint:
            wrapper = ET.Element(
                "body",
                {
                    "name": f"{prefix}root",
                    "pos": f"{float(offset_x):.9g} 0 0",
                },
            )
            for child in original_world_children:
                cloned = copy.deepcopy(child)
                prefix_local_names(cloned, prefix)
                wrapper.append(cloned)
            worldbody.append(wrapper)
            continue

        for child in original_world_children:
            cloned = copy.deepcopy(child)
            prefix_local_names(cloned, prefix)

            if not (cloned.tag == "body" and xml_subtree_has_freejoint(cloned)):
                offset_xml_pos(cloned, float(offset_x))
            worldbody.append(cloned)

    actuator = root.find("actuator")
    if actuator is not None:
        original_actuators = [copy.deepcopy(child) for child in list(actuator)]
        for child in list(actuator):
            actuator.remove(child)
        for copy_idx in range(n_copies):
            prefix = f"view{copy_idx}_"
            for child in original_actuators:
                cloned = copy.deepcopy(child)
                prefix_local_names(cloned, prefix)
                actuator.append(cloned)

    for tag in ("sensor", "equality", "contact", "keyframe"):
        remove_if_present(root, tag)

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix=".visualize_side_by_side_",
        dir=str(xml_path.parent),
        delete=False,
        encoding="utf-8",
    )
    with handle:
        handle.write(ET.tostring(root, encoding="unicode"))

    return Path(handle.name)


def convert_state_to_qpos_qvel_checked(
    state: np.ndarray,
    *,
    nq: int,
    nv: int,
    dof_per_entity: int,
    qpos_dim_per_entity: int | None = None,
    qvel_dim_per_entity: int | None = None,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a saved flat state to MuJoCo qpos/qvel and check dimensions."""
    state = np.asarray(state, dtype=np.float64)
    qpos, qvel = state_to_qpos_qvel(
        state,
        dof_per_entity=dof_per_entity,
        qpos_dim_per_entity=qpos_dim_per_entity,
        qvel_dim_per_entity=qvel_dim_per_entity,
    )
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)

    if qpos.shape != (nq,):
        raise ValueError(
            f"{label}: converted qpos has shape {qpos.shape}, expected {(nq,)}. "
            "Check that the trajectory file matches the MJCF and qpos/qvel dimensions."
        )
    if qvel.shape != (nv,):
        raise ValueError(
            f"{label}: converted qvel has shape {qvel.shape}, expected {(nv,)}. "
            "Check that the trajectory file matches the MJCF and qpos/qvel dimensions."
        )

    return qpos, qvel


def model_has_freejoints(model: mujoco.MjModel) -> bool:
    """Return whether a compiled MuJoCo model has any free joints."""
    return bool(np.any(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)))


def offset_qpos_x(qpos: np.ndarray, *, qpos_dim_per_entity: int, offset_x: float) -> np.ndarray:
    """Offset the world-x qpos entry of each entity.

    The public trajectory layouts store x as the first qpos coordinate per
    entity, including the 6-DOF drones whose per-entity qpos is
    [x, y, z, qw, qx, qy, qz].
    """
    if abs(float(offset_x)) <= 0.0:
        return qpos
    shifted = np.array(qpos, dtype=np.float64, copy=True)
    shifted[0::int(qpos_dim_per_entity)] += float(offset_x)
    return shifted


def set_frame_state(
    *,
    data: mujoco.MjData,
    base_model: mujoco.MjModel,
    states: Sequence[np.ndarray],
    labels: Sequence[str],
    dof_per_entity: int,
    qpos_dim_per_entity: int | None = None,
    qvel_dim_per_entity: int | None = None,
    qpos_x_offsets: Sequence[float] | None = None,
) -> None:
    """Write one replay frame into MuJoCo data."""
    qpos_parts: list[np.ndarray] = []
    qvel_parts: list[np.ndarray] = []

    if qpos_x_offsets is None:
        qpos_x_offsets = [0.0] * len(states)
    if len(qpos_x_offsets) != len(states):
        raise ValueError("qpos_x_offsets must match the number of replayed states.")

    qpos_dim = dof_per_entity if qpos_dim_per_entity is None else int(qpos_dim_per_entity)

    for state, label, offset_x in zip(states, labels, qpos_x_offsets, strict=False):
        qpos, qvel = convert_state_to_qpos_qvel_checked(
            state,
            nq=base_model.nq,
            nv=base_model.nv,
            dof_per_entity=dof_per_entity,
            qpos_dim_per_entity=qpos_dim_per_entity,
            qvel_dim_per_entity=qvel_dim_per_entity,
            label=label,
        )
        qpos = offset_qpos_x(qpos, qpos_dim_per_entity=qpos_dim, offset_x=float(offset_x))
        qpos_parts.append(qpos)
        qvel_parts.append(qvel)

    qpos = np.concatenate(qpos_parts, axis=0)
    qvel = np.concatenate(qvel_parts, axis=0)

    if qpos.shape != data.qpos.shape:
        raise ValueError(f"Display qpos has shape {qpos.shape}, expected {data.qpos.shape}.")
    if qvel.shape != data.qvel.shape:
        raise ValueError(f"Display qvel has shape {qvel.shape}, expected {data.qvel.shape}.")

    data.qpos[:] = qpos
    data.qvel[:] = qvel

    if data.ctrl.size:
        data.ctrl[:] = 0.0


def format_cost(dataset: TrajectorySet, rollout_idx: int) -> str:
    """Format a rollout cost for console output."""
    if dataset.costs is None:
        return "cost=n/a"
    return f"cost={float(dataset.costs[rollout_idx]):.6g}"


def print_clip_info(
    *,
    rollout_idx: int,
    primary: TrajectorySet,
    compare: TrajectorySet | None,
    t_frames: int,
) -> None:
    """Print the current clip label and cost information."""
    if compare is None:
        print(
            f"Playing rollout {rollout_idx:04d} | {primary.label}: "
            f"{format_cost(primary, rollout_idx)} | frames={t_frames}"
        )
    else:
        print(
            f"Playing rollout {rollout_idx:04d} | left {primary.label}: "
            f"{format_cost(primary, rollout_idx)} | right {compare.label}: "
            f"{format_cost(compare, rollout_idx)} | frames={t_frames}"
        )


def configure_camera(viewer: Any, *, compare: bool, spacing: float, top_down: bool) -> None:
    """Set a top-down viewer camera when requested."""
    if not top_down:
        return

    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])
    viewer.cam.distance = max(10.0, 1.75 * spacing if compare else 8.0)
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -90.0


def mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    """Return a MuJoCo object name, or an empty string."""
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return name or ""


def material_or_object_rgba(
    model: mujoco.MjModel,
    *,
    mat_id: int,
    object_rgba: np.ndarray,
) -> np.ndarray:
    """Return material color when present, otherwise the object color."""
    if int(mat_id) >= 0:
        rgba = np.array(model.mat_rgba[int(mat_id)], dtype=np.float32)
    else:
        rgba = np.array(object_rgba, dtype=np.float32)

    if rgba.shape != (4,):
        rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

    return rgba


def geom_effective_rgba(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Return the visible RGBA color for a geom."""
    return material_or_object_rgba(
        model,
        mat_id=int(model.geom_matid[int(geom_id)]),
        object_rgba=np.asarray(model.geom_rgba[int(geom_id)]),
    )


def site_effective_rgba(model: mujoco.MjModel, site_id: int) -> np.ndarray:
    """Return the visible RGBA color for a site."""
    return material_or_object_rgba(
        model,
        mat_id=int(model.site_matid[int(site_id)]),
        object_rgba=np.asarray(model.site_rgba[int(site_id)]),
    )


def rgba_saturation(rgb: np.ndarray) -> float:
    """Return max-minus-min channel saturation."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return float(np.max(rgb) - np.min(rgb))


def rgba_brightness(rgb: np.ndarray) -> float:
    """Return mean RGB brightness."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return float(np.mean(rgb))


def fallback_agent_colors(alpha: float) -> list[np.ndarray]:
    """Return a small palette for traces when model colors are unclear."""
    return [
        np.array([0.10, 0.40, 1.00, alpha], dtype=np.float32),
        np.array([1.00, 0.25, 0.10, alpha], dtype=np.float32),
        np.array([0.10, 0.80, 0.25, alpha], dtype=np.float32),
        np.array([0.80, 0.10, 1.00, alpha], dtype=np.float32),
        np.array([1.00, 0.70, 0.10, alpha], dtype=np.float32),
        np.array([0.00, 0.80, 0.90, alpha], dtype=np.float32),
        np.array([0.95, 0.20, 0.55, alpha], dtype=np.float32),
        np.array([0.45, 0.75, 0.10, alpha], dtype=np.float32),
    ]


def colors_are_distinct(colors: Sequence[np.ndarray], *, threshold: float = 0.16) -> bool:
    """Return whether inferred RGB colors are separated enough to be useful."""
    if len(colors) <= 1:
        return True
    rgbs = [np.asarray(color, dtype=np.float32)[:3] for color in colors]
    for i, rgb_i in enumerate(rgbs):
        for rgb_j in rgbs[i + 1 :]:
            if float(np.linalg.norm(rgb_i - rgb_j)) < threshold:
                return False
    return True


def task_marker_colors(model: mujoco.MjModel, *, n_agents: int, alpha: float) -> list[np.ndarray] | None:
    """Infer per-agent colors from task start/goal markers when bodies are identical."""
    colors: list[np.ndarray] = []
    for agent_idx in range(n_agents):
        marker_names = (
            f"start_d{agent_idx + 1}",
            f"goal_d{agent_idx + 1}_core",
            f"goal_d{agent_idx + 1}",
            f"agent{agent_idx + 1}_marker",
        )
        color = None
        for marker_name in marker_names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, marker_name)
            if geom_id < 0:
                continue
            rgba = geom_effective_rgba(model, geom_id)
            if visual_color_score(
                rgba=rgba,
                name=marker_name,
                size_score=geom_size_score(model, geom_id),
                is_site=False,
            ) > -1.0:
                color = np.array(rgba, dtype=np.float32, copy=True)
                color[3] = float(alpha)
                break
        if color is None:
            return None
        colors.append(color)

    if not colors_are_distinct(colors):
        return None
    return colors


def body_descendants(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    """Return a body and all descendants in MuJoCo body order."""
    root_body_id = int(root_body_id)
    descendants = [root_body_id]

    for body_id in range(root_body_id + 1, model.nbody):
        parent = int(model.body_parentid[body_id])
        if parent in descendants:
            descendants.append(body_id)

    return descendants


def geom_size_score(model: mujoco.MjModel, geom_id: int) -> float:
    """Return a simple visual-size score for a geom."""
    size = np.asarray(model.geom_size[int(geom_id)], dtype=np.float32)
    return float(np.linalg.norm(size))


def site_size_score(model: mujoco.MjModel, site_id: int) -> float:
    """Return a simple visual-size score for a site."""
    size = np.asarray(model.site_size[int(site_id)], dtype=np.float32)
    return float(np.linalg.norm(size))


def name_penalty(name: str) -> float:
    """Score down names that usually describe helpers or dark parts."""
    name_l = name.lower()

    penalty = 0.0

    weak_terms = (
        "shadow",
        "floor",
        "ground",
        "wall",
        "glass",
        "window",
        "windshield",
        "obstacle",
        "obst",
        "marker",
        "zone",
        "goal",
        "start",
        "headlight",
        "taillight",
        "light",
        "bumper",
        "trim",
        "hub",
        "rim",
    )
    for term in weak_terms:
        if term in name_l:
            penalty += 2.0

    # These are often dark/grey parts attached to an otherwise colored agent.
    dark_part_terms = (
        "rotor",
        "prop",
        "wheel",
        "tire",
        "arm",
        "leg",
        "dark",
        "thruster",
        "thrust",
        "motor",
    )
    for term in dark_part_terms:
        if term in name_l:
            penalty += 0.75

    return penalty


def name_bonus(name: str) -> float:
    """Score up names that usually identify the main agent body."""
    name_l = name.lower()

    bonus = 0.0
    strong_terms = (
        "agent",
        "body",
        "main",
        "core",
        "shell",
        "drone",
        "robot",
        "car",
        "sphere",
        "ball",
        "capsule",
        "torso",
        "nose",
    )
    for term in strong_terms:
        if term in name_l:
            bonus += 0.5

    return bonus


def visual_color_score(
    *,
    rgba: np.ndarray,
    name: str,
    size_score: float,
    is_site: bool,
) -> float:
    """Score how likely a color is to be useful for an agent trace."""
    rgba = np.asarray(rgba, dtype=np.float32)
    rgb = rgba[:3]
    alpha = float(rgba[3])

    sat = rgba_saturation(rgb)
    bright = rgba_brightness(rgb)

    if alpha <= 0.001:
        return -1e9

    score = 0.0
    score += 6.0 * sat
    score += 1.5 * alpha
    score += 0.5 * min(size_score, 1.0)

    if bright < 0.12:
        score -= 2.0
    if bright > 0.92 and sat < 0.10:
        score -= 2.0

    if sat < 0.08:
        score -= 3.0

    score += name_bonus(name)
    score -= name_penalty(name)

    if is_site:
        score -= 0.35

    return float(score)


def joint_qpos_order(model: mujoco.MjModel) -> list[int]:
    """Return joint ids sorted by qpos address."""
    joint_ids = list(range(model.njnt))
    joint_ids.sort(key=lambda j: int(model.jnt_qposadr[j]))
    return joint_ids


def joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    """Return the qpos width for one MuJoCo joint."""
    joint_type = int(model.jnt_type[int(joint_id)])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def infer_agent_body_groups(model: mujoco.MjModel, *, qpos_dim_per_entity: int) -> list[list[int]]:
    """Group bodies by each agent's qpos slice."""
    if qpos_dim_per_entity <= 0:
        raise ValueError("--qpos_dim_per_entity must be positive.")

    ordered_joints = joint_qpos_order(model)
    groups: list[list[int]] = []

    for start in range(0, model.nq, qpos_dim_per_entity):
        end = start + qpos_dim_per_entity
        joint_group = [
            joint_id
            for joint_id in ordered_joints
            if int(model.jnt_qposadr[joint_id]) < end
            and int(model.jnt_qposadr[joint_id]) + joint_qpos_width(model, joint_id) > start
        ]
        if not joint_group:
            continue

        body_ids: list[int] = []
        for joint_id in joint_group:
            body_id = int(model.jnt_bodyid[int(joint_id)])
            if body_id not in body_ids:
                body_ids.append(body_id)

        expanded: list[int] = []
        for body_id in body_ids:
            for descendant in body_descendants(model, body_id):
                if descendant not in expanded:
                    expanded.append(descendant)

        groups.append(expanded)

    return groups


def candidate_colors_for_body_group(
    model: mujoco.MjModel,
    body_ids: Sequence[int],
) -> list[tuple[float, np.ndarray, str]]:
    """Collect scored color candidates from geoms and sites in a body group."""
    candidates: list[tuple[float, np.ndarray, str]] = []

    body_id_set = {int(body_id) for body_id in body_ids}

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id not in body_id_set:
            continue

        name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        rgba = geom_effective_rgba(model, geom_id)
        score = visual_color_score(
            rgba=rgba,
            name=name,
            size_score=geom_size_score(model, geom_id),
            is_site=False,
        )
        candidates.append((score, rgba, f"geom:{name or geom_id}"))

    for site_id in range(model.nsite):
        body_id = int(model.site_bodyid[site_id])
        if body_id not in body_id_set:
            continue

        name = mj_name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        rgba = site_effective_rgba(model, site_id)
        score = visual_color_score(
            rgba=rgba,
            name=name,
            size_score=site_size_score(model, site_id),
            is_site=True,
        )
        candidates.append((score, rgba, f"site:{name or site_id}"))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def infer_agent_colors_from_model(
    model: mujoco.MjModel,
    *,
    n_agents: int,
    qpos_dim_per_entity: int,
    alpha: float,
    verbose: bool = False,
) -> list[np.ndarray]:
    """Infer one trace color per agent from the MJCF."""
    body_groups = infer_agent_body_groups(model, qpos_dim_per_entity=qpos_dim_per_entity)
    fallback = fallback_agent_colors(alpha)

    colors: list[np.ndarray] = []
    sources: list[str] = []
    scores: list[float] = []

    for agent_idx in range(n_agents):
        if agent_idx < len(body_groups):
            candidates = candidate_colors_for_body_group(model, body_groups[agent_idx])
        else:
            candidates = []

        if candidates and candidates[0][0] > -1.0:
            score, rgba, source = candidates[0]
            color = np.array(rgba, dtype=np.float32, copy=True)
            color[3] = float(alpha)
            colors.append(color)
            sources.append(source)
            scores.append(score)
        else:
            color = np.array(fallback[agent_idx % len(fallback)], dtype=np.float32, copy=True)
            color[3] = float(alpha)
            colors.append(color)
            sources.append("fallback palette")
            scores.append(float("nan"))

    if not colors_are_distinct(colors):
        marker_colors = task_marker_colors(model, n_agents=n_agents, alpha=alpha)
        if marker_colors is not None:
            colors = marker_colors
            sources = [f"task marker {idx + 1}" for idx in range(n_agents)]
            scores = [float("nan")] * n_agents
        else:
            colors = [np.array(fallback[idx % len(fallback)], dtype=np.float32, copy=True) for idx in range(n_agents)]
            for color in colors:
                color[3] = float(alpha)
            sources = ["fallback palette"] * n_agents
            scores = [float("nan")] * n_agents

    if verbose:
        for agent_idx, color in enumerate(colors):
            score = scores[agent_idx]
            score_text = "" if np.isnan(score) else f", score={score:.3f}"
            print(
                f"Trace color agent {agent_idx}: "
                f"rgba={np.round(color, 4).tolist()} from {sources[agent_idx]}{score_text}"
            )

    return colors


def layout_dims(args: argparse.Namespace) -> tuple[int, int, int]:
    """Resolve qpos and qvel dimensions from CLI arguments."""
    dof = int(args.dof_per_entity)
    qpos_dim = dof if args.qpos_dim_per_entity is None else int(args.qpos_dim_per_entity)
    qvel_dim = dof if args.qvel_dim_per_entity is None else int(args.qvel_dim_per_entity)
    if dof <= 0 or qpos_dim <= 0 or qvel_dim <= 0:
        raise ValueError("State layout dimensions must be positive.")
    return dof, qpos_dim, qvel_dim


def trajectory_xy_points(
    traj: np.ndarray,
    *,
    base_model: mujoco.MjModel,
    dof_per_entity: int,
    qpos_dim_per_entity: int | None = None,
    qvel_dim_per_entity: int | None = None,
    offset_x: float = 0.0,
) -> np.ndarray:
    """Convert a flat trajectory into per-agent trace points."""
    points = []

    for state in traj:
        qpos, _ = convert_state_to_qpos_qvel_checked(
            state,
            nq=base_model.nq,
            nv=base_model.nv,
            dof_per_entity=dof_per_entity,
            qpos_dim_per_entity=qpos_dim_per_entity,
            qvel_dim_per_entity=qvel_dim_per_entity,
            label="trace",
        )

        qpos_dim = dof_per_entity if qpos_dim_per_entity is None else int(qpos_dim_per_entity)
        if qpos.size % qpos_dim != 0:
            raise ValueError(
                f"qpos size {qpos.size} is not divisible by qpos_dim_per_entity={qpos_dim}."
            )

        qpos_by_entity = qpos.reshape(-1, qpos_dim)
        if qpos_dim >= 7:
            pos = qpos_by_entity[:, :3].copy()
            pos[:, 0] += offset_x
        else:
            xy = qpos_by_entity[:, :2].copy()
            xy[:, 0] += offset_x
            pos = xy
        points.append(pos)

    return np.asarray(points, dtype=np.float64)


def clear_mjv_geom_label(geom: Any) -> None:
    """Clear optional MuJoCo visual-geom text labels.

    Offscreen rendering reuses renderer.scene geoms.  mjv_connector updates the
    connector geometry but does not reliably clear every text/label field in all
    MuJoCo builds.  When a recycled geom slot previously contains non-zero label
    bytes, the trace capsules can be accompanied by random glyphs in the video.
    This helper is deliberately defensive across MuJoCo Python versions.
    """
    try:
        geom.label = ""
        return
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        geom.label = b""
        return
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        geom.label[:] = b"\0" * len(geom.label)
    except (AttributeError, TypeError, ValueError):
        pass


def initialize_connector_geom(geom: Any, *, width: float, rgba: np.ndarray) -> None:
    """Put a visual geom slot in a clean state before mjv_connector fills it."""
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        np.array([width, width, width], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    clear_mjv_geom_label(geom)


def draw_trace_geoms(
    scene: Any,
    traces: Sequence[np.ndarray],
    *,
    frame_idx: int,
    width: float,
    stride: int,
    alpha: float,
    z: float,
    agent_colors: Sequence[np.ndarray],
    reset_scene: bool,
) -> None:
    """Draw rollout traces into either a viewer user scene or a renderer scene."""
    if reset_scene:
        scene.ngeom = 0

    stride = max(1, int(stride))
    width = max(1e-6, float(width))

    if not agent_colors:
        agent_colors = fallback_agent_colors(alpha)

    for trace in traces:
        last = min(frame_idx + 1, trace.shape[0])

        for agent_idx in range(trace.shape[1]):
            color = np.array(
                agent_colors[agent_idx % len(agent_colors)],
                dtype=np.float32,
                copy=True,
            )
            color[3] = float(alpha)

            pts = trace[:last:stride, agent_idx]

            for a, b in zip(pts[:-1], pts[1:]):
                if scene.ngeom >= scene.maxgeom:
                    return

                geom = scene.geoms[scene.ngeom]

                if a.shape[0] >= 3:
                    from_pt = np.array([float(a[0]), float(a[1]), float(a[2])], dtype=np.float64)
                    to_pt = np.array([float(b[0]), float(b[1]), float(b[2])], dtype=np.float64)
                else:
                    from_pt = np.array([float(a[0]), float(a[1]), float(z)], dtype=np.float64)
                    to_pt = np.array([float(b[0]), float(b[1]), float(z)], dtype=np.float64)

                initialize_connector_geom(geom, width=width, rgba=color)

                mujoco.mjv_connector(
                    geom,
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                    width,
                    from_pt,
                    to_pt,
                )
                clear_mjv_geom_label(geom)

                geom.matid = -1
                geom.rgba[:] = color

                scene.ngeom += 1


def draw_traces(
    viewer,
    traces: Sequence[np.ndarray],
    *,
    frame_idx: int,
    width: float,
    stride: int,
    alpha: float,
    z: float,
    agent_colors: Sequence[np.ndarray],
) -> None:
    """Draw rollout traces in the viewer user scene."""
    draw_trace_geoms(
        viewer.user_scn,
        traces,
        frame_idx=frame_idx,
        width=width,
        stride=stride,
        alpha=alpha,
        z=z,
        agent_colors=agent_colors,
        reset_scene=True,
    )


def recording_requested(args: argparse.Namespace) -> bool:
    """Return whether this invocation should render MP4s instead of opening the GUI."""
    return args.record_dir is not None or args.record_path is not None


def safe_filename(value: str) -> str:
    """Create a readable, portable filename stem from a label."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return stem or "trajectory"


def record_fps(model: mujoco.MjModel, args: argparse.Namespace) -> float:
    """Resolve video FPS from CLI or model timestep."""
    if args.record_fps is not None:
        fps = float(args.record_fps)
    else:
        fps = float(args.playback_speed) / float(model.opt.timestep)
    if fps <= 0.0:
        raise ValueError("--record_fps must be positive when provided.")
    return fps


def video_path_for_dataset(
    args: argparse.Namespace,
    dataset: TrajectorySet,
    *,
    rollout_idx: int | None,
    multiple_datasets: bool,
) -> Path:
    """Resolve the output path for one dataset/rollout video."""
    label_stem = safe_filename(dataset.label)

    if args.record_path is not None:
        base = Path(args.record_path)
        suffix = base.suffix or ".mp4"
        parent = base.parent if str(base.parent) else Path(".")
        if multiple_datasets or args.record_split_rollouts:
            parts = [base.stem, label_stem]
            if rollout_idx is not None:
                parts.append(f"rollout{rollout_idx:04d}")
            return parent / ("_".join(parts) + suffix)
        return parent / (base.stem + suffix)

    directory = Path(args.record_dir or "videos")
    if args.record_split_rollouts and rollout_idx is not None:
        return directory / f"{label_stem}_rollout{rollout_idx:04d}.mp4"
    return directory / f"{label_stem}.mp4"


def make_renderer(model: mujoco.MjModel, args: argparse.Namespace) -> mujoco.Renderer:
    """Create a MuJoCo offscreen renderer with trace-friendly geom capacity."""
    try:
        return mujoco.Renderer(
            model,
            height=int(args.record_height),
            width=int(args.record_width),
            max_geom=int(args.record_maxgeom),
        )
    except TypeError:
        return mujoco.Renderer(
            model,
            height=int(args.record_height),
            width=int(args.record_width),
        )


def smoothstep(s: float) -> float:
    """Cubic smoothstep for camera interpolation."""
    s = min(1.0, max(0.0, float(s)))
    return s * s * (3.0 - 2.0 * s)


def free_camera_for_record_frame(
    model: mujoco.MjModel,
    *,
    frame_idx: int,
    n_frames: int,
    args: argparse.Namespace,
) -> mujoco.MjvCamera:
    """Create a deterministic free camera for one exported frame."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    center = np.asarray(model.stat.center, dtype=np.float64).reshape(3)
    extent = max(float(model.stat.extent), 1.0)
    cam.lookat[:] = center
    cam.distance = float(args.record_distance) if args.record_distance is not None else max(8.0, 1.55 * extent)

    mode = str(args.record_camera)
    if mode == "top_down":
        cam.azimuth = 90.0
        cam.elevation = -90.0
    elif mode == "side":
        cam.azimuth = 0.0
        cam.elevation = float(args.record_elevation)
    elif mode == "angled":
        cam.azimuth = float(args.record_orbit_start)
        cam.elevation = float(args.record_elevation)
    else:
        s = smoothstep(frame_idx / max(n_frames - 1, 1))
        cam.azimuth = float(args.record_orbit_start) + float(args.record_orbit_degrees) * s
        cam.elevation = float(args.record_elevation) + float(args.record_elevation_wobble) * np.sin(np.pi * s)

    return cam


def record_camera_for_frame(
    model: mujoco.MjModel,
    *,
    frame_idx: int,
    n_frames: int,
    args: argparse.Namespace,
) -> Any:
    """Return either a fixed-camera name or a scripted free camera."""
    if str(args.record_camera) == "fixed":
        return str(args.record_camera_name)
    return free_camera_for_record_frame(model, frame_idx=frame_idx, n_frames=n_frames, args=args)


def overlay_video_title(
    frame: np.ndarray,
    *,
    title: str,
    rollout_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Draw a simple title band into an RGB frame."""
    if args.record_no_titles:
        return frame

    title_height = int(args.record_title_height)
    if title_height <= 0:
        return frame

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    width, _height = img.size
    band_h = min(max(title_height, 24), max(24, img.size[1] // 3))

    draw.rectangle((0, 0, width, band_h), fill=(255, 255, 255))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, int(0.44 * band_h)))
        small_font = ImageFont.truetype("DejaVuSans.ttf", max(10, int(0.28 * band_h)))
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((18, max(4, int(0.12 * band_h))), title, fill=(0, 0, 0), font=font)
    draw.text(
        (18, max(18, int(0.62 * band_h))),
        f"rollout {rollout_idx:04d}",
        fill=(80, 80, 80),
        font=small_font,
    )
    return np.asarray(img)


def open_video_writer(path: Path, *, fps: float):
    """Open an imageio MP4 writer with a clear error if ffmpeg support is missing."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video export requires imageio with ffmpeg support. Install it with, for example, "
            "`uv add imageio imageio-ffmpeg` or `pip install imageio imageio-ffmpeg`."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path),
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


def validate_recording_args(args: argparse.Namespace) -> None:
    """Validate video-export arguments before allocating renderers."""
    if int(args.record_width) <= 0 or int(args.record_height) <= 0:
        raise ValueError("--record_width and --record_height must be positive.")
    if int(args.record_maxgeom) <= 0:
        raise ValueError("--record_maxgeom must be positive.")
    if args.loop and args.num_rollouts is None:
        raise ValueError("Recording with --loop requires --num_rollouts to avoid an infinite export.")
    if args.playback_speed <= 0.0:
        raise ValueError("--playback_speed must be > 0.")
    if args.pause_between_rollouts < 0.0:
        raise ValueError("--pause_between_rollouts must be >= 0.")


def write_recorded_clip(
    *,
    writer: Any,
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dataset: TrajectorySet,
    rollout_idx: int,
    args: argparse.Namespace,
    dof_per_entity: int,
    qpos_dim_per_entity: int,
    qvel_dim_per_entity: int,
    agent_colors: Sequence[np.ndarray],
) -> np.ndarray | None:
    """Render one rollout from one dataset to an already-open writer."""
    traj = dataset.trajectories[rollout_idx]
    t_frames = int(traj.shape[0])
    labels = [dataset.label]

    if args.show_traces:
        trace_points = [
            trajectory_xy_points(
                traj,
                base_model=model,
                dof_per_entity=dof_per_entity,
                qpos_dim_per_entity=qpos_dim_per_entity,
                qvel_dim_per_entity=qvel_dim_per_entity,
                offset_x=0.0,
            )
        ]
    else:
        trace_points = []

    last_frame: np.ndarray | None = None

    for frame_idx in range(t_frames):
        set_frame_state(
            data=data,
            base_model=model,
            states=[traj[frame_idx]],
            labels=labels,
            dof_per_entity=dof_per_entity,
            qpos_dim_per_entity=qpos_dim_per_entity,
            qvel_dim_per_entity=qvel_dim_per_entity,
            qpos_x_offsets=[0.0],
        )
        mujoco.mj_forward(model, data)

        camera = record_camera_for_frame(
            model,
            frame_idx=frame_idx,
            n_frames=t_frames,
            args=args,
        )
        renderer.update_scene(data, camera=camera)

        if args.show_traces:
            draw_trace_geoms(
                renderer.scene,
                trace_points,
                frame_idx=frame_idx,
                width=float(args.trace_width),
                stride=int(args.trace_stride),
                alpha=float(args.trace_alpha),
                z=float(args.trace_z),
                agent_colors=agent_colors,
                reset_scene=False,
            )

        frame = renderer.render()
        frame = overlay_video_title(
            frame,
            title=dataset.label,
            rollout_idx=rollout_idx,
            args=args,
        )
        writer.append_data(frame)
        last_frame = frame

    return last_frame


def record_dataset_to_path(
    *,
    path: Path,
    model: mujoco.MjModel,
    dataset: TrajectorySet,
    rollout_ids: Sequence[int],
    args: argparse.Namespace,
    dof_per_entity: int,
    qpos_dim_per_entity: int,
    qvel_dim_per_entity: int,
    agent_colors: Sequence[np.ndarray],
    fps: float,
) -> None:
    """Render one dataset to one MP4, possibly with multiple rollouts concatenated."""
    data = mujoco.MjData(model)
    pause_frames = int(round(float(args.pause_between_rollouts) * fps))

    with make_renderer(model, args) as renderer:
        with open_video_writer(path, fps=fps) as writer:
            for pos, rollout_idx in enumerate(rollout_ids):
                if not args.quiet:
                    print(f"Recording {dataset.label} rollout {rollout_idx:04d} -> {path}")
                last_frame = write_recorded_clip(
                    writer=writer,
                    renderer=renderer,
                    model=model,
                    data=data,
                    dataset=dataset,
                    rollout_idx=int(rollout_idx),
                    args=args,
                    dof_per_entity=dof_per_entity,
                    qpos_dim_per_entity=qpos_dim_per_entity,
                    qvel_dim_per_entity=qvel_dim_per_entity,
                    agent_colors=agent_colors,
                )
                if (
                    last_frame is not None
                    and pause_frames > 0
                    and pos + 1 < len(rollout_ids)
                    and not args.record_split_rollouts
                ):
                    for _ in range(pause_frames):
                        writer.append_data(last_frame)


def record_videos(
    *,
    base_model: mujoco.MjModel,
    primary: TrajectorySet,
    compare: TrajectorySet | None,
    args: argparse.Namespace,
) -> None:
    """Record one separate video per trajectory set using the original one-copy MJCF."""
    validate_recording_args(args)

    n_available = validate_compatible_sets(primary, compare)
    rollout_ids = list(
        rollout_indices(
            n_rollouts=n_available,
            start_idx=int(args.rollout_idx),
            selection=args.selection,
            loop=bool(args.loop),
            num_rollouts=args.num_rollouts,
            seed=int(args.seed),
        )
    )
    if not rollout_ids:
        raise ValueError("No rollout indices were selected for recording.")

    dof_per_entity, qpos_dim_per_entity, qvel_dim_per_entity = layout_dims(args)
    if base_model.nq % qpos_dim_per_entity != 0:
        raise ValueError(
            f"base_model.nq={base_model.nq} is not divisible by "
            f"--qpos_dim_per_entity={qpos_dim_per_entity}."
        )
    if base_model.nv % qvel_dim_per_entity != 0:
        raise ValueError(
            f"base_model.nv={base_model.nv} is not divisible by "
            f"--qvel_dim_per_entity={qvel_dim_per_entity}."
        )

    n_agents = base_model.nq // qpos_dim_per_entity
    n_vel_entities = base_model.nv // qvel_dim_per_entity
    if n_agents != n_vel_entities:
        raise ValueError(
            f"qpos layout implies {n_agents} entities, but qvel layout implies {n_vel_entities}."
        )

    agent_colors = infer_agent_colors_from_model(
        base_model,
        n_agents=n_agents,
        qpos_dim_per_entity=qpos_dim_per_entity,
        alpha=float(args.trace_alpha),
        verbose=bool(args.print_trace_colors),
    )

    datasets = [primary] if compare is None else [primary, compare]
    fps = record_fps(base_model, args)
    multiple_datasets = len(datasets) > 1

    print(
        f"Recording {len(rollout_ids)} rollout(s) at {fps:.3g} fps, "
        f"{int(args.record_width)}x{int(args.record_height)}, camera={args.record_camera}."
    )

    if args.record_split_rollouts:
        for rollout_idx in rollout_ids:
            for dataset in datasets:
                path = video_path_for_dataset(
                    args,
                    dataset,
                    rollout_idx=int(rollout_idx),
                    multiple_datasets=multiple_datasets,
                )
                record_dataset_to_path(
                    path=path,
                    model=base_model,
                    dataset=dataset,
                    rollout_ids=[int(rollout_idx)],
                    args=args,
                    dof_per_entity=dof_per_entity,
                    qpos_dim_per_entity=qpos_dim_per_entity,
                    qvel_dim_per_entity=qvel_dim_per_entity,
                    agent_colors=agent_colors,
                    fps=fps,
                )
        return

    for dataset in datasets:
        path = video_path_for_dataset(
            args,
            dataset,
            rollout_idx=None,
            multiple_datasets=multiple_datasets,
        )
        record_dataset_to_path(
            path=path,
            model=base_model,
            dataset=dataset,
            rollout_ids=rollout_ids,
            args=args,
            dof_per_entity=dof_per_entity,
            qpos_dim_per_entity=qpos_dim_per_entity,
            qvel_dim_per_entity=qvel_dim_per_entity,
            agent_colors=agent_colors,
            fps=fps,
        )


def replay(
    *,
    base_model: mujoco.MjModel,
    display_model: mujoco.MjModel,
    primary: TrajectorySet,
    compare: TrajectorySet | None,
    args: argparse.Namespace,
) -> None:
    """Replay one or two trajectory sets in a MuJoCo viewer."""
    if args.playback_speed <= 0.0:
        raise ValueError("--playback_speed must be > 0.")
    if args.pause_between_rollouts < 0.0:
        raise ValueError("--pause_between_rollouts must be >= 0.")
    if args.trace_alpha < 0.0 or args.trace_alpha > 1.0:
        raise ValueError("--trace_alpha must be between 0 and 1.")
    if args.trace_stride <= 0:
        raise ValueError("--trace_stride must be positive.")
    if args.trace_width <= 0.0:
        raise ValueError("--trace_width must be positive.")

    data = mujoco.MjData(display_model)
    frame_dt = float(display_model.opt.timestep) / float(args.playback_speed)

    n_available = validate_compatible_sets(primary, compare)
    indices = rollout_indices(
        n_rollouts=n_available,
        start_idx=int(args.rollout_idx),
        selection=args.selection,
        loop=bool(args.loop),
        num_rollouts=args.num_rollouts,
        seed=int(args.seed),
    )

    labels = [primary.label] if compare is None else [primary.label, compare.label]

    dof_per_entity, qpos_dim_per_entity, qvel_dim_per_entity = layout_dims(args)

    if base_model.nq % qpos_dim_per_entity != 0:
        raise ValueError(
            f"base_model.nq={base_model.nq} is not divisible by "
            f"--qpos_dim_per_entity={qpos_dim_per_entity}."
        )
    if base_model.nv % qvel_dim_per_entity != 0:
        raise ValueError(
            f"base_model.nv={base_model.nv} is not divisible by "
            f"--qvel_dim_per_entity={qvel_dim_per_entity}."
        )
    n_agents = base_model.nq // qpos_dim_per_entity
    n_vel_entities = base_model.nv // qvel_dim_per_entity
    if n_agents != n_vel_entities:
        raise ValueError(
            f"qpos layout implies {n_agents} entities, but qvel layout implies {n_vel_entities}."
        )

    # When comparing free-joint models, copies cannot be offset by nesting them
    # under translated wrapper bodies, so shift the world-x qpos entries here.
    if compare is not None and model_has_freejoints(base_model):
        qpos_display_offsets = [
            -0.5 * float(args.side_by_side_spacing),
            0.5 * float(args.side_by_side_spacing),
        ]
    else:
        qpos_display_offsets = [0.0] if compare is None else [0.0, 0.0]

    agent_colors = infer_agent_colors_from_model(
        base_model,
        n_agents=n_agents,
        qpos_dim_per_entity=qpos_dim_per_entity,
        alpha=float(args.trace_alpha),
        verbose=bool(args.print_trace_colors),
    )

    with mujoco.viewer.launch_passive(display_model, data) as viewer:
        configure_camera(
            viewer,
            compare=compare is not None,
            spacing=float(args.side_by_side_spacing),
            top_down=bool(args.top_down),
        )

        for rollout_idx in indices:
            if not viewer.is_running():
                return

            primary_traj = primary.trajectories[rollout_idx]

            if compare is None:
                clip_trajs = [primary_traj]
            else:
                compare_traj = compare.trajectories[rollout_idx]
                clip_trajs = [primary_traj, compare_traj]

            t_frames = min(int(traj.shape[0]) for traj in clip_trajs)

            if not args.quiet:
                print_clip_info(
                    rollout_idx=rollout_idx,
                    primary=primary,
                    compare=compare,
                    t_frames=t_frames,
                )

            if args.show_traces:
                if compare is None:
                    offsets = [0.0]
                else:
                    offsets = [
                        -0.5 * float(args.side_by_side_spacing),
                        0.5 * float(args.side_by_side_spacing),
                    ]

                trace_points = [
                    trajectory_xy_points(
                        traj,
                        base_model=base_model,
                        dof_per_entity=dof_per_entity,
                        qpos_dim_per_entity=qpos_dim_per_entity,
                        qvel_dim_per_entity=qvel_dim_per_entity,
                        offset_x=offset,
                    )
                    for traj, offset in zip(clip_trajs, offsets)
                ]
            else:
                trace_points = []

            for frame_idx in range(t_frames):
                if not viewer.is_running():
                    return

                wall_start = time.time()

                states = [traj[frame_idx] for traj in clip_trajs]

                set_frame_state(
                    data=data,
                    base_model=base_model,
                    states=states,
                    labels=labels,
                    dof_per_entity=dof_per_entity,
                    qpos_dim_per_entity=qpos_dim_per_entity,
                    qvel_dim_per_entity=qvel_dim_per_entity,
                    qpos_x_offsets=qpos_display_offsets,
                )

                mujoco.mj_forward(display_model, data)

                if args.show_traces:
                    draw_traces(
                        viewer,
                        trace_points,
                        frame_idx=frame_idx,
                        width=float(args.trace_width),
                        stride=int(args.trace_stride),
                        alpha=float(args.trace_alpha),
                        z=float(args.trace_z),
                        agent_colors=agent_colors,
                    )
                else:
                    viewer.user_scn.ngeom = 0

                viewer.sync()

                elapsed = time.time() - wall_start
                remaining = frame_dt - elapsed

                if remaining > 0.0:
                    time.sleep(remaining)

            if args.pause_between_rollouts > 0.0:
                time.sleep(float(args.pause_between_rollouts))


def print_dataset_summary(primary: TrajectorySet, compare: TrajectorySet | None, n_available: int) -> None:
    """Print loaded dataset shapes before opening the viewer."""
    print(
        f"Loaded {primary.label}: {primary.n_rollouts} rollouts, "
        f"T={primary.t_end}, state_dim={primary.state_dim}."
    )
    if compare is not None:
        print(
            f"Loaded {compare.label}: {compare.n_rollouts} rollouts, "
            f"T={compare.t_end}, state_dim={compare.state_dim}."
        )
        print(f"Loaded comparison set; {n_available} shared rollout indices are available.")


def dry_run(
    *,
    xml_path: Path,
    primary: TrajectorySet,
    compare: TrajectorySet | None,
    args: argparse.Namespace,
    n_available: int,
) -> None:
    """Validate replay inputs without opening MuJoCo's viewer."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    dof_per_entity, qpos_dim_per_entity, qvel_dim_per_entity = layout_dims(args)
    if model.nq % qpos_dim_per_entity != 0:
        raise ValueError(
            f"Model nq={model.nq} is not divisible by "
            f"--qpos_dim_per_entity={qpos_dim_per_entity}."
        )
    if model.nv % qvel_dim_per_entity != 0:
        raise ValueError(
            f"Model nv={model.nv} is not divisible by "
            f"--qvel_dim_per_entity={qvel_dim_per_entity}."
        )
    n_agents = model.nq // qpos_dim_per_entity
    rollout_idx = int(args.rollout_idx)
    if rollout_idx < 0 or rollout_idx >= n_available:
        raise IndexError(f"rollout_idx={rollout_idx} is out of bounds for {n_available} rollouts.")

    datasets = [primary] if compare is None else [primary, compare]
    for dataset in datasets:
        traj = dataset.trajectories[rollout_idx]
        for frame_idx in (0, max(0, dataset.t_end - 1)):
            qpos, qvel = convert_state_to_qpos_qvel_checked(
                traj[frame_idx],
                nq=model.nq,
                nv=model.nv,
                dof_per_entity=dof_per_entity,
                qpos_dim_per_entity=qpos_dim_per_entity,
                qvel_dim_per_entity=qvel_dim_per_entity,
                label=f"{dataset.label} frame {frame_idx}",
            )
            if qpos.shape[0] != model.nq or qvel.shape[0] != model.nv:
                raise ValueError(
                    f"{dataset.path} frame {frame_idx} maps to qpos/qvel "
                    f"{qpos.shape[0]}/{qvel.shape[0]}, expected {model.nq}/{model.nv}."
                )
        if bool(args.show_traces) or bool(args.print_trace_colors):
            trajectory_xy_points(
                traj,
                base_model=model,
                dof_per_entity=dof_per_entity,
                qpos_dim_per_entity=qpos_dim_per_entity,
                qvel_dim_per_entity=qvel_dim_per_entity,
                offset_x=0.0,
            )

    infer_agent_colors_from_model(
        model,
        n_agents=n_agents,
        qpos_dim_per_entity=qpos_dim_per_entity,
        alpha=float(args.trace_alpha),
        verbose=bool(args.print_trace_colors),
    )
    print(
        f"Dry run OK: nq={model.nq}, nv={model.nv}, n_agents={n_agents}, "
        f"qpos/entity={qpos_dim_per_entity}, qvel/entity={qvel_dim_per_entity}."
    )


def main() -> None:
    """Load trajectory files and start replay."""
    args = parse_args()
    xml_path = Path(args.xml_path)

    primary = load_trajectory_set(args.trajectories_path, args.label)
    compare = (
        load_trajectory_set(args.compare_trajectories_path, args.compare_label)
        if args.compare_trajectories_path is not None
        else None
    )

    n_available = validate_compatible_sets(primary, compare)
    print_dataset_summary(primary, compare, n_available)

    if args.dry_run:
        dry_run(
            xml_path=xml_path,
            primary=primary,
            compare=compare,
            args=args,
            n_available=n_available,
        )
        return

    if recording_requested(args):
        temp_record_xml_path: Path | None = None
        try:
            temp_record_xml_path = build_recording_xml(
                xml_path,
                offwidth=int(args.record_width),
                offheight=int(args.record_height),
            )
            record_model = mujoco.MjModel.from_xml_path(str(temp_record_xml_path))
            record_videos(
                base_model=record_model,
                primary=primary,
                compare=compare,
                args=args,
            )
        finally:
            if temp_record_xml_path is not None:
                try:
                    os.remove(temp_record_xml_path)
                except OSError:
                    pass
        return

    base_model = mujoco.MjModel.from_xml_path(str(xml_path))

    temp_xml_path: Path | None = None

    try:
        if compare is None:
            display_model = base_model
        else:
            temp_xml_path = build_side_by_side_xml(
                xml_path,
                spacing=float(args.side_by_side_spacing),
                n_copies=2,
            )
            display_model = mujoco.MjModel.from_xml_path(str(temp_xml_path))

            expected_nq = 2 * base_model.nq
            expected_nv = 2 * base_model.nv

            if display_model.nq != expected_nq or display_model.nv != expected_nv:
                raise ValueError(
                    "The side-by-side display model did not duplicate the base model's DOFs as expected: "
                    f"got nq={display_model.nq}, nv={display_model.nv}, "
                    f"expected nq={expected_nq}, nv={expected_nv}."
                )

        replay(
            base_model=base_model,
            display_model=display_model,
            primary=primary,
            compare=compare,
            args=args,
        )

    finally:
        if temp_xml_path is not None:
            try:
                os.remove(temp_xml_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
