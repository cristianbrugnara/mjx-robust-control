# System Configuration

A supported system is a MJCF model plus one JSON config:

- `assets/mjcf/<system>.xml`: MuJoCo bodies, joints, actuators, geoms, cameras, simulation options, and any visual element.
- `assets/config/<system>.json`: state layout, controller inputs, cost terms, disturbances, metrics, and training default parameters.

## Compatibility

- The XML should load with `mujoco.MjModel.from_xml_path` and `mjx.put_model`.
- `n_agents * qpos_dim_per_entity` must match the `qpos_idx` length.
- `n_agents * qvel_dim_per_entity` must match the `qvel_idx` length.
- If `qpos_idx` or `qvel_idx` is not provided, the full MuJoCo `qpos` or `qvel` vector is used.
- The flat state is `[agent_0 qpos, agent_0 qvels, agent_1 qpos, agent_1 qvel, ..., agent_n qpos, agent_n qvel]`.
- Control dimensions, policy bounds, labels, `x0`, `xbar`, `q_diag_per_entity`, and noise masks must have the configured lengths.
- Cost terms, metrics, obstacles, bounds, references, and disturbances must match the configured state layout.

To check:

```bash
python src/check_compatibility.py \
  --xml_path assets/mjcf/<system>.xml \
  --system_config_path assets/config/<system>.json
```

## Controller Inputs

`task.controller_inputs` is concatenated in JSON order. Block types:

- `state`
- `state_error`
- `imc_residual`

The default and recommended configuration feeds the controller only the
`imc_residual` block, so the learned operator acts on the internal-model
residual alone. This keeps the controller in the pure IMC form, where
closed-loop stability is preserved by construction regardless of the learned
weights.

Additional blocks such as `state` or `state_error` may be concatenated when a
task benefits from explicit tracking-error feedback. These introduce a direct
state-feedback path, so their magnitude should be kept small (via `scale`) to
stay within a comfortable stability margin. Example:

```json
"controller_inputs": [
  {
    "type": "state_error",
    "scale": 0.1,
    "clip": null,
    "params": { "target": "xbar", "sign": "current_minus_target" }
  },
  {
    "type": "imc_residual",
    "scale": 1.0,
    "clip": null,
    "params": {}
  }
]
```

## Cost Terms

`task.cost_terms` defines scalar rollout costs. Each term has `type`, `weight`, `where`, and `params`.

- `state_l2`
- `state_bounds`
- `control_l2`
- `pairwise_distance_barrier`
- `ellipsoid_obstacle`
- `box_bounds`

### Control Interfaces

- `direct_actuator`: policy controls map directly to MuJoCo actuators.
- `quadrotor_attitude_mixer`: `[collective_thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]` maps to four rotor thrust actuators.
- `quadrotor_wrench_mixer`: `[collective_thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]` maps to body thrust and body moments.

### Supported Systems

### `corridor`

- XML: `assets/mjcf/corridor.xml`
- JSON: `assets/config/corridor.json`
- State: two point-mass agents, per-agent `[qx, qy, vx, vy]`
- Interface: `direct_actuator`
- Main costs: `state_l2`, `control_l2`, `pairwise_distance_barrier`, `ellipsoid_obstacle`, `box_bounds`

### `drones3_3d`

- XML: `assets/mjcf/drones3_3d.xml`
- JSON: `assets/config/drones3_3d.json`
- State: three free-joint quadrotors, per-drone `[x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]`
- Interface: `quadrotor_attitude_mixer`
- Main costs: `state_l2`, `control_l2`, `pairwise_distance_barrier`, `ellipsoid_obstacle`, `box_bounds`

### `crazyflies3_3d`

- XML: `assets/mjcf/crazyflies3_3d.xml`
- JSON: `assets/config/crazyflies3_3d.json`
- State: three free-joint Crazyflie models, per-drone `[x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]`
- Interface: `quadrotor_attitude_mixer`
- Main costs: `state_l2`, `control_l2`, `pairwise_distance_barrier`, `ellipsoid_obstacle`, `box_bounds`
