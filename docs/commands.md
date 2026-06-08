# Commands

Run commands from the repository root. With conda, activate the environment and use the commands. With UV, use `uv run python`, e.g.
`uv run python src/train.py ...`.

## Generic System

Use this template for a compatible MJCF file and JSON system configuration.
Replace `<system_name>`, `<objective>`, and any dimensions or batch sizes that
are specific to the system.

```bash
# Check compatibility first.
python src/check_compatibility.py \
  --xml_path assets/mjcf/<system_name>.xml \
  --system_config_path assets/config/<system_name>.json

# Train controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/<system_name>.xml \
  --system_config_path assets/config/<system_name>.json \
  --save_path artifacts/trained_models/<system_name>_<objective>_seed3.eqx \
  --objective <objective> \
  --seed 3 \
  --epochs 12000 \
  --n_train 512 \
  --n_valid 512 \
  --batch_size 16 \
  --validation_period 50 \
  --resample_train_batch \
  --control_squash \
  --grad_clip_norm 1.0

# Optional controller-size overrides:
#   --n_xi_override 32 \
#   --l_override 32

# Optional initial-condition/output overrides:
#   --std_ini_override 0.0 \
#   --std_ini_param 0.0 \
#   --output_amplification 0.0

# Save prestabilizer only.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/<system_name>.xml \
  --system_config_path assets/config/<system_name>.json \
  --save_path artifacts/trained_models/<system_name>_only_stab_seed3.eqx \
  --objective mean \
  --seed 3 \
  --epochs 1 \
  --learning_rate 0.0 \
  --n_train 1 \
  --n_valid 1 \
  --batch_size 1 \
  --validation_period 1 \
  --std_ini_override 0.0 \
  --std_ini_param 0.0 \
  --output_amplification 0.0 \
  --control_squash

# Evaluate controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/evaluate.py \
  --xml_path assets/mjcf/<system_name>.xml \
  --checkpoint_path artifacts/trained_models/<system_name>_<objective>_seed3.eqx \
  --system_config_path assets/config/<system_name>.json \
  --n_rollouts 100 \
  --eval_batch_size 10 \
  --seed 3 \
  --output_dir artifacts/eval/<system_name>_<objective>_seed3

# Visualize rollouts.
python src/visualize.py \
  --xml_path assets/mjcf/<system_name>.xml \
  --trajectories_path artifacts/eval/<system_name>_<objective>_seed3/trajectories.npy \
  --qpos_dim_per_entity <qpos_dim_per_entity> \
  --qvel_dim_per_entity <qvel_dim_per_entity> \
  --selection sequential \
  --show_traces \
  --playback_speed 1.0 \
  --loop


# Certify objective checkpoints. Use any subset of the listed objectives.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/certify.py \
  --xml_path assets/mjcf/<system_name>.xml \
  --sys_model <system_name> \
  --system_config_path assets/config/<system_name>.json \
  --objectives mean cvar pinball softmax worst_case \
  --seed 3 \
  --checkpoint_dir artifacts/trained_models \
  --output_dir artifacts/certification \
  --run_name <system_name>_seed3 \
  --m_cert 200 \
  --n_eval 100
```

Common parameters to adjust:

| Parameter                                            | Description                                                                         |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `<system_name>`                                      | Checkpoint filename prefix used by certification.                                   |
| `<objective>`                                        | One of `mean`, `cvar`, `pinball`, `softmax`, or `worst_case`.                       |
| `--epochs`                                           | Increase for harder systems.                                                        |
| `--batch_size`                                       | Use a value that fits memory.                                                       |
| `--n_xi_override`, `--l_override`                    | Optional REN controller-size overrides.                                             |
| `--n_rollouts`, `--eval_batch_size`                  | Evaluation sample count and evaluation memory tradeoff.                             |
| `--qpos_dim_per_entity`, `--qvel_dim_per_entity`     | Required by visualization so it can split trajectories per entity.                  |
| `--m_cert`, `--n_eval`                               | Certification calibration size and held-out evaluation size.                        |

## Corridor

```bash
# Check compatibility.
python src/check_compatibility.py \
  --xml_path assets/mjcf/corridor.xml \
  --system_config_path assets/config/corridor.json

# Train controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/corridor.xml \
  --system_config_path assets/config/corridor.json \
  --save_path artifacts/trained_models/corridor_<objective>_seed3.eqx \
  --objective <objective> \
  --seed 3 \
  --epochs 5000 \
  --n_train 512 \
  --n_valid 512 \
  --batch_size 100 \
  --validation_period 50 \
  --control_squash \
  --grad_clip_norm 1.0

# Save prestabilizer.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/corridor.xml \
  --system_config_path assets/config/corridor.json \
  --save_path artifacts/trained_models/corridor_only_stab_seed3.eqx \
  --objective mean \
  --seed 3 \
  --epochs 1 \
  --learning_rate 0.0 \
  --n_train 1 \
  --n_valid 1 \
  --batch_size 1 \
  --validation_period 1 \
  --std_ini_override 0.0 \
  --std_ini_param 0.0 \
  --output_amplification 0.0 \
  --control_squash

# Evaluate controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/evaluate.py \
  --xml_path assets/mjcf/corridor.xml \
  --checkpoint_path artifacts/trained_models/corridor_<objective>_seed3.eqx \
  --system_config_path assets/config/corridor.json \
  --n_rollouts 100 \
  --eval_batch_size 100 \
  --seed 3 \
  --output_dir artifacts/eval/corridor_<objective>_seed3

# Visualize rollouts.
python src/visualize.py \
  --xml_path assets/mjcf/corridor.xml \
  --trajectories_path artifacts/eval/corridor_<objective>_seed3/trajectories.npy \
  --qpos_dim_per_entity 2 \
  --qvel_dim_per_entity 2 \
  --selection sequential \
  --show_traces \
  --top_down \
  --playback_speed 0.85 \
  --loop

# Certify controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/certify.py \
  --xml_path assets/mjcf/corridor.xml \
  --sys_model corridor \
  --system_config_path assets/config/corridor.json \
  --objectives mean cvar pinball softmax worst_case \
  --seed 3 \
  --checkpoint_dir artifacts/trained_models \
  --output_dir artifacts/certification \
  --run_name corridor_seed3 \
  --m_cert 200 \
  --n_eval 100
```

## Intersection3

```bash
# Check compatibility.
python src/check_compatibility.py \
  --xml_path assets/mjcf/intersection3_actual_cars.xml \
  --system_config_path assets/config/intersection3.json

# Train controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/intersection3_actual_cars.xml \
  --system_config_path assets/config/intersection3.json \
  --save_path artifacts/trained_models/intersection3_<objective>_seed3.eqx \
  --objective <objective> \
  --seed 3 \
  --epochs 12000 \
  --n_train 512 \
  --n_valid 512 \
  --batch_size 16 \
  --validation_period 50 \
  --resample_train_batch \
  --n_xi_override 32 \
  --l_override 32 \
  --control_squash \
  --grad_clip_norm 1.0

# Save prestabilizer.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/intersection3_actual_cars.xml \
  --system_config_path assets/config/intersection3.json \
  --save_path artifacts/trained_models/intersection3_only_stab_seed3.eqx \
  --objective mean \
  --seed 3 \
  --epochs 1 \
  --learning_rate 0.0 \
  --n_train 1 \
  --n_valid 1 \
  --batch_size 1 \
  --validation_period 1 \
  --std_ini_override 0.0 \
  --std_ini_param 0.0 \
  --output_amplification 0.0 \
  --n_xi_override 32 \
  --l_override 32 \
  --control_squash

# Evaluate controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/evaluate.py \
  --xml_path assets/mjcf/intersection3_actual_cars.xml \
  --checkpoint_path artifacts/trained_models/intersection3_<objective>_seed3.eqx \
  --system_config_path assets/config/intersection3.json \
  --n_rollouts 64 \
  --eval_batch_size 8 \
  --seed 3 \
  --output_dir artifacts/eval/intersection3_<objective>_seed3

# Visualize rollouts.
python src/visualize.py \
  --xml_path assets/mjcf/intersection3_actual_cars.xml \
  --trajectories_path artifacts/eval/intersection3_<objective>_seed3/trajectories.npy \
  --qpos_dim_per_entity 3 \
  --qvel_dim_per_entity 3 \
  --selection sequential \
  --show_traces \
  --top_down \
  --playback_speed 1.25 \
  --loop

# Certify controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/certify.py \
  --xml_path assets/mjcf/intersection3_actual_cars.xml \
  --sys_model intersection3 \
  --system_config_path assets/config/intersection3.json \
  --objectives mean cvar pinball softmax worst_case \
  --seed 3 \
  --checkpoint_dir artifacts/trained_models \
  --output_dir artifacts/certification \
  --run_name intersection3_seed3 \
  --m_cert 200 \
  --n_eval 100
```

## Drones3 3D

```bash
# Check compatibility.
python src/check_compatibility.py \
  --xml_path assets/mjcf/drones3_3d.xml \
  --system_config_path assets/config/drones3_3d.json

# Train controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/drones3_3d.xml \
  --system_config_path assets/config/drones3_3d.json \
  --save_path artifacts/trained_models/drones3_3d_<objective>_seed3.eqx \
  --objective <objective> \
  --seed 3 \
  --epochs 12000 \
  --n_train 512 \
  --n_valid 512 \
  --batch_size 6 \
  --validation_period 50 \
  --resample_train_batch \
  --control_squash \
  --grad_clip_norm 1.0

# Save prestabilizer.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/drones3_3d.xml \
  --system_config_path assets/config/drones3_3d.json \
  --save_path artifacts/trained_models/drones3_3d_only_stab_seed3.eqx \
  --objective mean \
  --seed 3 \
  --epochs 1 \
  --learning_rate 0.0 \
  --n_train 1 \
  --n_valid 1 \
  --batch_size 1 \
  --validation_period 1 \
  --std_ini_override 0.0 \
  --std_ini_param 0.0 \
  --output_amplification 0.0 \
  --control_squash

# Evaluate controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/evaluate.py \
  --xml_path assets/mjcf/drones3_3d.xml \
  --checkpoint_path artifacts/trained_models/drones3_3d_<objective>_seed3.eqx \
  --system_config_path assets/config/drones3_3d.json \
  --n_rollouts 24 \
  --eval_batch_size 3 \
  --seed 3 \
  --output_dir artifacts/eval/drones3_3d_<objective>_seed3

# Visualize rollouts.
python src/visualize.py \
  --xml_path assets/mjcf/drones3_3d.xml \
  --trajectories_path artifacts/eval/drones3_3d_<objective>_seed3/trajectories.npy \
  --qpos_dim_per_entity 7 \
  --qvel_dim_per_entity 6 \
  --selection sequential \
  --show_traces \
  --playback_speed 0.65 \
  --loop

# Certify controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/certify.py \
  --xml_path assets/mjcf/drones3_3d.xml \
  --sys_model drones3_3d \
  --system_config_path assets/config/drones3_3d.json \
  --objectives mean cvar pinball softmax worst_case \
  --seed 3 \
  --checkpoint_dir artifacts/trained_models \
  --output_dir artifacts/certification \
  --run_name drones3_3d_seed3 \
  --m_cert 200 \
  --n_eval 100
```

## Crazyflies3 3D

```bash
# Check compatibility.
python src/check_compatibility.py \
  --xml_path assets/mjcf/crazyflies3_3d.xml \
  --system_config_path assets/config/crazyflies3_3d.json

# Train controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/crazyflies3_3d.xml \
  --system_config_path assets/config/crazyflies3_3d.json \
  --save_path artifacts/trained_models/crazyflies3_3d_<objective>_seed3.eqx \
  --objective <objective> \
  --seed 3 \
  --epochs 12000 \
  --n_train 512 \
  --n_valid 512 \
  --batch_size 6 \
  --validation_period 50 \
  --resample_train_batch \
  --control_squash \
  --grad_clip_norm 1.0

# Save prestabilizer.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/train.py \
  --xml_path assets/mjcf/crazyflies3_3d.xml \
  --system_config_path assets/config/crazyflies3_3d.json \
  --save_path artifacts/trained_models/crazyflies3_3d_only_stab_seed3.eqx \
  --objective mean \
  --seed 3 \
  --epochs 1 \
  --learning_rate 0.0 \
  --n_train 1 \
  --n_valid 1 \
  --batch_size 1 \
  --validation_period 1 \
  --std_ini_override 0.0 \
  --std_ini_param 0.0 \
  --output_amplification 0.0 \
  --control_squash

# Evaluate controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/evaluate.py \
  --xml_path assets/mjcf/crazyflies3_3d.xml \
  --checkpoint_path artifacts/trained_models/crazyflies3_3d_<objective>_seed3.eqx \
  --system_config_path assets/config/crazyflies3_3d.json \
  --n_rollouts 24 \
  --eval_batch_size 3 \
  --seed 3 \
  --output_dir artifacts/eval/crazyflies3_3d_<objective>_seed3

# Visualize rollouts. Crazyflie drones are small so traces should be scaled down
python src/visualize.py \
  --xml_path assets/mjcf/crazyflies3_3d.xml \
  --trajectories_path artifacts/eval/crazyflies3_3d_<objective>_seed3/trajectories.npy \
  --qpos_dim_per_entity 7 \
  --qvel_dim_per_entity 6 \
  --selection sequential \
  --show_traces \
  --trace_width 0.004 \
  --trace_z 0.012 \
  --playback_speed 0.65 \
  --loop

# Certify controller.
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
python src/certify.py \
  --xml_path assets/mjcf/crazyflies3_3d.xml \
  --sys_model crazyflies3_3d \
  --system_config_path assets/config/crazyflies3_3d.json \
  --objectives mean cvar pinball softmax worst_case \
  --seed 3 \
  --checkpoint_dir artifacts/trained_models \
  --output_dir artifacts/certification \
  --run_name crazyflies3_3d_seed3 \
  --m_cert 200 \
  --n_eval 100
```
