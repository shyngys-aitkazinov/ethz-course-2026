# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ETH Zurich Robot Learning HW3: Imitation learning with the SO-101 robot arm in MuJoCo simulation. The pipeline involves teleoperation data collection, action computation, policy training (action-chunking MLP), and evaluation. Three exercises: single-cube with obstacle (MSE policy), DAgger for out-of-distribution recovery, and multicube goal-conditioned policy.

## Setup

```bash
cd hw3_imitation_learning
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Key Commands

```bash
# Configure teleoperation keys (produces hw3/keymap.json)
python scripts/configure_keys.py

# Record teleoperation demos (adds .zarr files to datasets/raw/single_cube/teleop/)
python scripts/record_teleop_demos.py
python scripts/record_teleop_demos.py --multicube  # for exercise 3

# Compute delta actions from raw recordings
python scripts/compute_actions.py --action-space ee       # 3D ee position deltas
python scripts/compute_actions.py --action-space ee_full  # 6D pos+euler deltas
python scripts/compute_actions.py --action-space joints   # 5D joint angle deltas
python scripts/compute_actions.py --action-space ee --datasets-dir ./datasets/raw/multi_cube

# Train policy
python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
    --state-keys state_ee_xyz state_gripper "state_cube[:3]" \
    --action-keys action_ee_xyz action_gripper --policy obstacle

# Evaluate policy
python scripts/eval.py --checkpoint <path.pt>
python scripts/eval.py --checkpoint <path.pt> --headless --num-episodes 100
python scripts/eval.py --checkpoint <path.pt> --adversarial  # ex2: shifted obstacle distribution
python scripts/eval.py --checkpoint <path.pt> --multicube     # ex3

# DAgger data collection (ex2)
python scripts/dagger_eval.py --checkpoint <path.pt>

# Generate submission files
python student_eval/run_eval --exercise 1 --checkpoint <path.pt>
```

## Architecture

- **`hw3/model.py`** — Policy definitions. `BasePolicy` (abstract) requires `compute_loss` and `sample_actions`. Students implement `ObstaclePolicy` (ex1/2, MSE loss MLP) and `MultiTaskPolicy` (ex3, goal-conditioned). `build_policy()` factory constructs them. Class names and default init args must match submitted checkpoints.
- **`hw3/dataset.py`** — `SO100ChunkDataset` returns `(state, action_chunk)` pairs. `Normalizer` does feature-wise z-normalization. `load_zarr`/`load_and_merge_zarrs` support key specs with column slicing (e.g. `"state_cube[:3]"`).
- **`hw3/eval_utils.py`** — Checkpoint loading, observation-to-state assembly, action application (delta-based for ee/joints, absolute for gripper), success checking.
- **`hw3/sim_env.py`** — MuJoCo simulation environments (`SO100SimEnv`, `SO100MulticubeSimEnv`).
- **`scripts/train.py`** — Training harness with TODOs for: hyperparameters, training step, eval step, optimizer/scheduler. Saves checkpoints with model weights, normalizer stats, and metadata to `checkpoints/`.
- **`scripts/compute_actions.py`** — Converts raw teleop recordings into delta actions (a_t = s_{t+1} - s_t). Gripper actions are recorded control commands, not deltas.

## Data Flow

1. `record_teleop_demos.py` → raw `.zarr` in `datasets/raw/{single,multi}_cube/teleop/`
2. `compute_actions.py` → processed `.zarr` in `datasets/processed/{single,multi}_cube/`
3. `train.py` reads processed zarr → saves `.pt` checkpoint to `checkpoints/`
4. DAgger episodes go to `datasets/raw/single_cube/dagger/`, then `compute_actions.py` merges teleop + dagger data

## State/Action Spaces

Actions are deltas between consecutive states. Three action spaces: `ee` (3D xyz), `ee_full` (6D pos+euler), `joints` (5D, excludes Jaw). Gripper is always a separate absolute control signal (`action_gripper`). State keys support slicing syntax: `"state_cube[:3]"` takes only position from the 7D cube state.

## Checkpoint Format

Saved as dict with keys: `model_state_dict`, `normalizer` (mean/std arrays), `chunk_size`, `policy_type`, `state_keys`, `action_keys`, `state_dim`, `action_dim`, `val_loss`, `epoch`.

## Constraints

- No additional library imports beyond what's in `pyproject.toml`
- Policy eval runs max 800 steps
- Don't modify `student_eval/run_eval.py` or `.hwresult` files
- `ObstaclePolicy` and `MultiTaskPolicy` class names are imported by autograder — don't rename
- Default constructor args must match the trained checkpoint for autograder reproduction
- Less than 1M parameters needed for 100% SR on ex1/ex2
