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

## Cost Terms

`task.cost_terms` defines scalar rollout costs. Each term has `type`, `weight`, `where`, and `params`.

- `state_l2`
- `state_bounds`
- `control_l2`
- `pairwise_distance_barrier`
- `ellipsoid_obstacle`
- `box_bounds`
- `road_network`
- `heading_to_goal`
- `planar_heading_velocity`

### Control Interfaces

- `direct_actuator`: policy controls map directly to MuJoCo actuators.
- `bicycle_steering`: `[drive_force, steering_angle]` policy controls map to car drive/yaw/lateral actuators.
- `quadrotor_attitude_mixer`: `[collective_thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]` maps to four rotor thrust actuators.
- `quadrotor_wrench_mixer`: `[collective_thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]` maps to body thrust and body moments.

### Supported Systems

### `corridor`

- XML: `assets/mjcf/corridor.xml`
- JSON: `assets/config/corridor.json`
- State: two point-mass agents, per-agent `[qx, qy, vx, vy]`
- Interface: `direct_actuator`
- Main costs: `state_l2`, `control_l2`, `pairwise_distance_barrier`, `ellipsoid_obstacle`, `box_bounds`

### `intersection3`

- XML: `assets/mjcf/intersection3_actual_cars.xml`
- JSON: `assets/config/intersection3.json`
- State: three planar cars, per-car `[x, y, yaw, vx, vy, omega]`
- Interface: `bicycle_steering`
- Main costs: `state_l2`, `control_l2`, `pairwise_distance_barrier`, `ellipsoid_obstacle`, `box_bounds`, `road_network`, `heading_to_goal`, `planar_heading_velocity`

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
