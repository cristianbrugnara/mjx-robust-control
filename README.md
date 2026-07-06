# Robust Neural Control with MuJoCo MJX

This repository is a JAX [3] port of the performance-boosting controllers at [DecodEPFL/performance-boosting_controllers](https://github.com/DecodEPFL/performance-boosting_controllers) [1],
extended with robust objective training [2] and MuJoCo/MJX systems [4]. It trains
neural feedback controllers through differentiable MJX rollouts, using
XML-defined environments (MJCF files) and JSON system configs for training, evaluation,
visualization, and probabilistic certification.

The system dynamics (ODEs) are not written explicitly in Python.
Instead, the physical dynamics are defined by the MuJoCo MJCF model and advanced by MJX.

Thus, a system is described by a MuJoCo MJCF model plus a JSON file that defines the
state layout, controller inputs, cost terms, and control interface.

The workflow follows four scripts:

1. `src/train.py`: train a performance-boosting controller using a robust objective, e.g., CVaR.
2. `src/evaluate.py`: run held-out rollouts and save trajectories.
3. `src/visualize.py`: replay saved trajectories in MuJoCo GUI.
4. `src/certify.py`: evaluate empirical certification thresholds.

<p align="center">
  <sub><em>Crazyflies GIF may take a few seconds to load.</em></sub>
</p>

<p align="center">
  <img src="https://github.com/cristianbrugnara/mjx-robust-control/releases/download/readme-media-v1/corridor_cvar.gif" alt="Corridor CVaR" width="45%">
  <img src="https://github.com/cristianbrugnara/mjx-robust-control/releases/download/readme-media-v1/crazyflies_cvar.gif" alt="Crazyflies CVaR" width="45%">
</p>

## Robust Objectives

Training minimizes a scalar rollout performance metric over randomized MJX rollouts. We denote one rollout by $D \sim p(D)$, the controller parameters by $K$, and the scalar rollout cost by $\Psi(D; K)$. Training uses a design set of sampled rollouts $\lbrace D_{\mathrm{des}}^{(i)} \rbrace_{i=1}^{m_{\mathrm{des}}}$, while certification uses an additional independent set of rollouts $\lbrace D_{\mathrm{cert}}^{(j)} \rbrace_{j=1}^{m_{\mathrm{cert}}}$.

Users select a target violation probability $\alpha \in (0,1)$ and confidence level $1-\delta$, with $\delta \in (0,1)$. The robust objectives below are intended to improve not only average performance, but also high-tail behavior.

Supported objectives:

### `mean`

Empirical average rollout cost:

```math
J_{\mathrm{mean}}(K)
=
\frac{1}{m}
\sum_{i=1}^{m}
\Psi(D^{(i)}; K).
```

### `cvar`

Rockafellar-Uryasev CVaR surrogate with trainable threshold $\tau$, targeting the worst $\alpha$-tail:

```math
J_{\mathrm{CVaR}}(K,\tau)
=
\tau
+
\frac{1}{\alpha m}
\sum_{i=1}^{m}
\max\left(0, \Psi(D^{(i)}; K)-\tau\right).
```

The corresponding training objective is

```math
\min_{K,\tau}
J_{\mathrm{CVaR}}(K,\tau).
```

### `pinball`

Quantile-oriented surrogate with trainable threshold $\tau$. For $q = 1-\alpha$:

```math
\rho_q(r)
=
q\max(0,r)
+
(1-q)\max(0,-r).
```

The empirical objective is

```math
\min_{K,\tau}
\frac{1}{m}
\sum_{i=1}^{m}
\rho_{1-\alpha}
\left(
\Psi(D^{(i)};K)-\tau
\right).
```

### `softmax`

Differentiable log-sum-exp surrogate for the empirical worst case:

```math
J_{\beta}(K)
=
\frac{1}{\beta}
\log
\left(
\sum_{i=1}^{m}
\exp\left(\beta \Psi(D^{(i)};K)\right)
\right),
```

where larger $\beta > 0$ makes the objective closer to the maximum rollout cost.

### `worst_case`

Empirical maximum rollout cost:

```math
J_{\mathrm{worst}}(K)
=
\max_{i=1,\ldots,m}
\Psi(D^{(i)};K).
```





Certification is performed after training using new held-out rollout costs, independent of the rollouts used for controller design. After evaluating the trained controller $K^\star$ on $m_{\mathrm{cert}}$ certification rollouts, the costs are sorted as

```math
\Psi^{(1)}
\leq
\Psi^{(2)}
\leq
\cdots
\leq
\Psi^{(m_{\mathrm{cert}})}.
```

Then 

```math
\epsilon_{m_{\mathrm{cert}}}
=
\sqrt{
\frac{\log(2/\delta)}
{2m_{\mathrm{cert}}}
},
```

and

```math
k^\star
=
\left\lceil
m_{\mathrm{cert}}
\left(
1-\alpha+\epsilon_{m_{\mathrm{cert}}}
\right)
\right\rceil.
```

If $k^\star \leq m_{\mathrm{cert}}$, the certified threshold is $\Psi^{(k^\star)}$. With confidence at least $1-\delta$, a new rollout $D_{\mathrm{new}}\sim p(D)$ satisfies

```math
\Pr\left(
\Psi(D_{\mathrm{new}};K^\star)
\leq
\Psi^{(k^\star)}
\right)
\geq
1-\alpha.
```


## Code Map

```
MJCF + JSON (assets/)
       │
       ▼
system_configs.py   ──►  workflow_utils.py
(SystemSpec, TaskSpec)    (build RolloutConfig,
                           sampling helpers,
                           metric helpers)
                               │
                               ▼
jax_models.py (REN)  ──►  jax_rollout.py  ◄──  jax_loss_functions.py
(PsiU, Controller)        (scan loop,            (state, collision,
                           control interfaces,    obstacle, barrier
                           step/terminal loss)    cost primitives)
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
             train.py               evaluate.py
             (build + optimise)     (held-out rollouts)
                   │                       │
        robust_objectives.py           certify.py
        (mean/CVaR/pinball/            (DKWM conformal certification)
         softmax/worst_case)               │
                                      visualize.py
                                      (GUI replay / video)
```

| File                       | Role                                                                              |
| -------------------------- | --------------------------------------------------------------------------------- |
| `system_configs.py`      | Frozen dataclasses for a system; JSON→Python parsing and MuJoCo validation       |
| `workflow_utils.py`      | Shared setup (build `RolloutConfig`, sampling, data initialization) and metrics |
| `jax_models.py`          | Acyclic REN controller (`PsiU`) with Lyapunov-stable parameter constraints      |
| `jax_rollout.py`         | Differentiable `lax.scan` rollout; cost term dispatch; control interfaces       |
| `jax_loss_functions.py`  | Primitive JAX cost functions (state tracking, collision, obstacle, barrier)       |
| `robust_objectives.py`   | Scalar objectives over a batch of rollout costs (mean, CVaR, pinball, …)         |
| `train.py`               | Training loop: vmapped rollouts → robust objective → Optax update               |
| `evaluate.py`            | Held-out evaluation; writes trajectories/costs/summary                            |
| `certify.py`             | Distribution-free certification via Theorem 1 (DKW confidence bound)              |
| `visualize.py`           | MuJoCo GUI replay with trace overlays; optional video export                      |
| `check_compatibility.py` | CLI: verify an XML + JSON pair loads cleanly and print resolved dims              |

## Currently Supported Systems

* `corridor`: two 2D point-mass agents navigating obstacle bars. 3D version of Corridor task from [1].
* `intersection3`: three cars crossing an intersection.
* `drones3_3d`: three 6-DOF quadrotors tracking 3D goals.
* `crazyflies3_3d`: similar settings to `drones3_3d`, but using Bitcraze Crazyflie 2 quadrotors from MuJoCo Menagerie [5] in the same 3D task.

For more details, refer to `docs/systems.md`.

## Environment

CPU is cheaper for development, tests, and visualization. GPU is recommended for training.

Clone with the MuJoCo Menagerie submodule, which provides the Crazyflie assets:

```bash
git clone --recurse-submodules git@github.com:cristianbrugnara/mjx-robust-control.git
```

### Conda

CPU:

```bash
conda env create -f env/conda-cpu.yml
conda activate jaxmjx
```

NVIDIA CUDA 12 GPU:

```bash
conda env create -f env/conda-gpu.yml
conda activate jaxmjx_gpu
```

### UV

CPU:

```bash
uv sync
```

NVIDIA CUDA 12 GPU:

```bash
uv sync --extra gpu
```

Use `uv run <command>`.


## Quick Start

Check that a system XML and JSON agree:

```bash
python src/check_compatibility.py \
  --xml_path assets/mjcf/corridor.xml \
  --system_config_path assets/config/corridor.json
```

Train a simple corridor controller using raw mean objective:

```bash
python src/train.py \
  --xml_path assets/mjcf/corridor.xml \
  --system_config_path assets/config/corridor.json \
  --save_path artifacts/trained_models/corridor_mean_seed3.eqx \
  --objective mean \
  --seed 3 \
  --epochs 5000 \
  --n_train 100 \
  --n_valid 100 \
  --batch_size 100 \
  --validation_period 50 \
  --control_squash \
  --grad_clip_norm 1.0
```

Evaluate the controller on new rollouts:

```bash
python src/evaluate.py \
  --xml_path assets/mjcf/corridor.xml \
  --checkpoint_path artifacts/trained_models/corridor_mean_seed3.eqx \
  --system_config_path assets/config/corridor.json \
  --n_rollouts 100 \
  --eval_batch_size 100 \
  --seed 3 \
  --output_dir artifacts/eval/corridor_mean_seed3
```

Visualize saved trajectories:

```bash
python src/visualize.py \
  --xml_path assets/mjcf/corridor.xml \
  --trajectories_path artifacts/eval/corridor_mean_seed3/trajectories.npy \
  --qpos_dim_per_entity 2 \
  --qvel_dim_per_entity 2 \
  --selection sequential \
  --show_traces \
  --top_down \
  --loop
```

Certify performance:

```bash
python src/certify.py \
  --xml_path assets/mjcf/corridor.xml \
  --system_config_path assets/config/corridor.json \
  --objectives mean \
  --seed 3 \
  --checkpoint_dir artifacts/trained_models \
  --output_dir artifacts/certification \
  --run_name corridor_seed3 \
  --m_cert 200 \
  --n_eval 100
```

Longer commands are in `docs/commands.md`.

## Outputs

Generated files are put in `artifacts/`:

* `artifacts/trained_models/`: `.eqx` checkpoints and `.meta.json` information.
* `artifacts/eval/`: rollout trajectories, controls, costs, and evaluation summaries.
* `artifacts/certification/`: certification arrays, selected examples, plots, and summaries.

## References

[1] L. Furieri, C. L. Galimberti, and G. Ferrari-Trecate, "Learning to Boost the Performance of Stable Nonlinear Systems," *IEEE Open J. Control Syst.*, vol. 3, pp. 342–357, 2024, doi: 10.1109/OJCSYS.2024.3441768.

[2] L. Meroi, S. El Amrani, D. Saccani, D. Piga, and G. Ferrari-Trecate, "Risk-Aware Stability-Preserving Design of Neural Controllers with Conformal Verification," unpublished report, 2026.

[3] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang, "JAX: composable transformations of Python+NumPy programs," 2018, version 0.2.5, http://github.com/google/jax.

[4] E. Todorov, T. Erez, and Y. Tassa, "MuJoCo: A physics engine for model-based control," in *2012 IEEE/RSJ Int. Conf. Intel. Robots Syst. (IROS)*, 2012, pp. 5026–5033, doi: 10.1109/IROS.2012.6386109.

[5] K. Zakka, Y. Tassa, and MuJoCo Menagerie Contributors, "MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo," 2022, [GitHub repository](http://github.com/google-deepmind/mujoco_menagerie).
