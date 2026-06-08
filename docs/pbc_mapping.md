# PBC To JAX/MJX Mapping

This is a high-level map from the original PyTorch PBC code under `PBC/` to
the JAX/MJX port in this repository. The port keeps the REN controller structure
but moves dynamics, rollout execution, task definitions, evaluation, and
visualization into a MuJoCo/MJX plus JSON-config workflow.

## Main Workflow

| PBC                                                          | JAX/MJX                                                              | Notes                                                                                                                                                 |
| ------------------------------------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PBC/run_corridor.py`                                      | `src/train.py`                                                      | Main training entry point. PBC trains a hard-coded corridor script; JAX/MJX uses `TrainConfig`, CLI flags, MuJoCo XML, and `SystemSpec` JSON.     |
| Training loop inside `run_corridor.py`                     | `train.train`, `train.build_loss_fn`, `train.make_train_step`   | The explicit Python time loop becomes vectorized/JIT-compatible rollout losses and Optax updates.                                                     |
| Validation block inside `run_corridor.py`                  | `train.make_eval_step`, validation logic in `train.train`         | Same role: evaluate held-out initial conditions and keep the best checkpoint.                                                                         |
| Final rollout and saved `.pt` model in `run_corridor.py` | `src/evaluate.py`, Equinox `.eqx` checkpoints plus `.meta.json` | Evaluation is split into a dedicated script that saves trajectories, controls, costs, and summaries.                                                  |
| `PBC/run_corridor_RNN.py`                                  | No direct public equivalent                                           | The port focuses on the stabilized acyclic REN controller, with optional no-stabilization checkpoints in artifacts rather than a separate RNN script. |
| `PBC/run_corridor_online_opt.py`                           | No direct equivalent                                                  | The CasADi/CVXPY online optimization baseline is not ported as a primary workflow.                                                                    |
| `PBC/run_wp.py`                                            | No direct public equivalent                                         | The public workflow is driven by the four JSON system configs listed in `docs/systems.md`.                                                           |

## Models And Controller

| PBC                                | JAX/MJX                                                                                      | Notes                                                                                                                                                                                               |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src.models.PsiU`                                | `src/jax_models.py::PsiU`                                                                  | Direct Equinox/JAX port of the acyclic REN. Free parameters remain `X`, `Y`, `B2`, `C2`, `D21`, `D22`, `D12`; derived constrained matrices are built in `_derived_matrices`.        |
| `PsiU.set_model_param`                           | `PsiU._derived_matrices`                                                                   | PBC stores derived matrices as mutable fields after optimizer steps; JAX/MJX recomputes them functionally during calls.                                                                             |
| `PsiU.forward`                                   | `PsiU.__call__`                                                                            | Same REN recurrence and output calculation, written with `jax.lax.fori_loop` and `jnp.linalg.solve`.                                                                                            |
| `src.models.PsiX`                                | `src/jax_models.py::PsiX`                                                                  | Same nominal one-step predictor wrapper.                                                                                                                                                            |
| `src.models.Input`                               | `src/jax_models.py::InputSchedule`                                                         | Optional trainable/feedforward schedule ported to a functional JAX module.                                                                                                                          |
| `src.models.Controller`                          | `src/jax_models.py::Controller`                                                            | Same IMC-style structure: form a model-error signal and pass it to `PsiU`. The port also exposes `step_from_signal` for rollout code that builds normalized controller inputs from JSON blocks. |
| `Controller.forward`                             | `Controller.__call__`, `step_from_omega`, `step_from_prediction`, `step_from_signal` | PBC has one forward path; JAX/MJX separates direct REN calls from rollout input construction.                                                                                                      |
| `ControllerRNN`, `PsiU_nonstab`                | No direct module-level equivalent                                                            | The public controller module contains the stabilized REN. Non-stabilized comparisons are represented by training settings/checkpoints rather than separate classes.                                  |

## Dynamics, Systems, And Data

| PBC                                       | JAX/MJX                                                                                                                                  | Notes                                                                                                                                                |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src.models.SystemRobots`                                   | `assets/mjcf/*.xml`, `assets/config/*.json`, `src/system_configs.py::SystemSpec`                                                 | The hard-coded point-mass plant becomes a MuJoCo model plus JSON state layout, costs, bounds, obstacles, and defaults.                               |
| `SystemRobots.f`                                            | `mjx.step` inside `src/jax_rollout.py::_scan_step`                                                                                   | Plant dynamics are no longer manually coded in Python for each system.                                                                                |
| `SystemRobots.forward`                                      | `src/jax_rollout.py::data_with_state`, `_scan_step`, `extract_flat_state`                                                          | Rollout state is stored in MJX data, then projected to the flat controller/cost state.                                                               |
| `SystemRobots.distance_agents`, `h_agents`                | `src/jax_loss_functions.py::f_loss_ca`, `src/workflow_utils.py::calculate_collisions`, pairwise distance cost terms in `jax_rollout.py` | Pairwise safety is generalized to configurable state layouts.                                                                                        |
| `SystemRobots.h_obstacles`, `distance_obstacle`           | `src/jax_loss_functions.py::f_loss_obst`, `src/workflow_utils.py::obstacle_ellipsoid_values`, `ellipsoid_obstacle` cost terms           | Obstacles are configured in JSON instead of hard-coded positions.                                                                                    |
| `src.utils.set_params`                                      | System JSON defaults plus `train.resolve_training_values`                                                                              | Training defaults such as horizon, weights, REN size, and noise scale live with each system config. CLI flags can override them.                     |
| `src.utils.get_ini_cond`                                    | `SystemSpec.x0_array`, `SystemSpec.xbar_array`                                                                                       | Nominal initial and target states are loaded from JSON.                                                                                              |
| `src.utils.generate_data`                                   | `train.sample_initial_conditions`, `evaluate.sample_initial_conditions`, `build_data_init`                                         | Initial-condition batches are sampled directly from PRNG keys rather than cached as pickle files.                                                    |
| `src.utils.count_parameters`                                | Equinox tree utilities where needed                                                                                                      | No dedicated helper in the public workflow.                                                                                                          |

## Rollouts And Losses

| PBC                                   | JAX/MJX                                                                                              | Notes                                                                                                                                        |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Inner time loop in `run_corridor.py`   | `src/jax_rollout.py::rollout`                                                                      | Uses `jax.lax.scan` over MJX steps and returns a scalar rollout cost.                                                                      |
| Rollout plus logging arrays              | `rollout_with_trajectory`                                                                       | Evaluation and visualization use saved states, policy controls, and costs.                                                                  |
| `src.loss_functions.f_loss_states`     | `src/jax_loss_functions.py::f_loss_states`, `jax_rollout.py` `state_l2` terms                  | Ported and extended with quaternion-aware state errors for 3D systems.                                                                       |
| `f_loss_u`                             | `src/jax_loss_functions.py::f_loss_u`, `jax_rollout.py` `control_l2` terms                     | Same quadratic control effort role.                                                                                                          |
| `f_loss_ca`                            | `src/jax_loss_functions.py::f_loss_ca`, `jax_rollout.py` `pairwise_distance_barrier` terms     | Same collision-avoidance idea, generalized by JSON layout and margins.                                                                       |
| `f_loss_obst`                          | `src/jax_loss_functions.py::f_loss_obst`, `jax_rollout.py` `ellipsoid_obstacle` terms          | PBC's Gaussian obstacle bumps become configurable ellipsoid penalties.                                                                       |
| `f_loss_side`                          | `src/jax_loss_functions.py::f_loss_side`, `jax_rollout.py` `box_bounds`/`state_bounds` terms | Bounds move from hard-coded corridor limits to JSON.                                                                                         |
| `src.loss_wp.*` temporal logic helpers | `jax_rollout.py` task cost terms and `system_configs.py::TaskSpec`                               | Not a one-for-one port. Public task logic is represented declaratively through references and cost terms.                                     |

## Robust Objectives And Certification

| PBC                                                                                                | JAX/MJX                                                  | Notes                                                                                     |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Mean rollout objective in `run_corridor.py`                                                      | `src/robust_objectives.py::mean_loss`                           | Default objective over a batch of rollout costs.                                          |
| CVaR, pinball, softmax, quantile, worst-case robust objectives from extended experiments/notebooks | `src/robust_objectives.py`                                      | JAX/MJX adds these as first-class objective choices used by `src/train.py`.             |
| Robustness checks over perturbed systems in PBC scripts                                            | `src/evaluate.py`, `src/certify.py`                           | Evaluation and empirical probabilistic certification are separated into reusable scripts. |
| Order-statistic certification logic                                                                | `src/certify.py::epsilon_m`, `k_star`, `theorem1_threshold` | New public certification layer around held-out rollout costs.                             |

## Plotting And Visualization

| PBC location                              | JAX/MJX location                                                                   | Notes                                                                                             |
| ----------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `src.plots.plot_trajectories`           | `src/visualize.py::replay`, `draw_traces`                                      | Matplotlib 2D plots become MuJoCo replay with optional traces.                                    |
| `src.plots.plot_traj_vs_time`           | Saved arrays from `src/evaluate.py`; custom plotting or `src/certify.py` plots | There is no exact time-series plotting helper in the main public workflow.                        |
| `src.plots_corridor*`, `src.plots_wp` | `src/visualize.py` plus system XML visuals                                       | Visualization is driven by MuJoCo models and saved trajectory files.                              |
| GIF/video artifacts                       | `src/visualize.py` replay path                                                   | The port focuses on interactive/replay visualization; saved video generation is not the main API. |

## Important Shape Changes

- PBC is script-centric and corridor-centric; JAX/MJX is config-centric and
  supports `corridor`, `intersection3`, `drones3_3d`, and `crazyflies3_3d`.
- PBC stores system constants in Python functions/classes; JAX/MJX stores most
  system semantics in `assets/config/*.json` and validates them against MJCF.
- PBC mutates PyTorch modules during training; JAX/MJX keeps controller modules
  as Equinox pytrees and uses Optax/JIT-friendly pure functions.
- PBC rolls out a hand-written plant; JAX/MJX rolls out MuJoCo dynamics through
  `mjx.step`.
- JAX/MJX adds robust objectives, checkpoint metadata, evaluation bundles,
  compatibility checks, and empirical certification as public workflow pieces.
