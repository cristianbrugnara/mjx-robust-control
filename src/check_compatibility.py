"""
Validation utility. Checks that an XML/JSON pair can be loaded by MJX, then prints resolved
state/control dimensions.
"""

import argparse

import mujoco
from mujoco import mjx

from system_configs import apply_mjx_model_options, load_system_spec


def check_system(xml_path: str, system_config_path: str) -> None:
    """Verify if XML/JSON-config pair can enter MJX."""
    spec = load_system_spec(system_config_path)
    spec.validate_basic()

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    spec.validate_against_mj_model(mj_model)
    apply_mjx_model_options(spec, mj_model)

    mjx_model = mjx.put_model(mj_model)
    _ = mjx.make_data(mjx_model)

    print("OK")
    print("nq:", mj_model.nq)
    print("nv:", mj_model.nv)
    print("nu:", mj_model.nu)
    print("qpos_idx:", spec.resolve_qpos_idx(mj_model.nq))
    print("qvel_idx:", spec.resolve_qvel_idx(mj_model.nv))
    print("qpos_dim_per_entity:", spec.qpos_dim_per_entity_resolved)
    print("qvel_dim_per_entity:", spec.qvel_dim_per_entity_resolved)
    print("state_dim:", spec.state_dim)
    print("control_dim:", spec.control_dim)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check MuJoCo XML and system config compatibility."
    )
    parser.add_argument(
        "--xml_path",
        required=True,
        help="Path to the MuJoCo XML file.",
    )
    parser.add_argument(
        "--system_config_path",
        required=True,
        help="Path to the system config file.",
    )

    args = parser.parse_args()

    check_system(
        xml_path=args.xml_path,
        system_config_path=args.system_config_path,
    )


if __name__ == "__main__":
    main()
