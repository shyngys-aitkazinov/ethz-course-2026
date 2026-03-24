# compute_actions.py — Explained

This script converts **raw teleoperation recordings** (`.zarr` stores) into **training-ready datasets** by computing delta actions between consecutive timesteps.

## Core Idea

The robot records states at each timestep during teleoperation. To train a policy, we need **actions** — which are defined as the difference between consecutive states:

```
a_t = s_{t+1} - s_{t}
```

The last timestep of each episode is dropped because there's no future state to compute a delta from.

**Exception:** Gripper actions are NOT computed as deltas. They are the raw control commands recorded during teleop (absolute values), because the gripper needs to push harder than the state reflects to actually grip the cube.

## Pipeline

```
datasets/raw/{single,multi}_cube/**/*.zarr
        │
        ▼
  load_and_merge_zarrs()    ← concatenate all .zarr stores, track episode boundaries
        │
        ▼
  select_action_space()     ← pick which state columns to use for action computation
        │
        ▼
  compute_actions_for_episodes()  ← compute s_{t+1} - s_t per episode
        │
        ▼
  trim_to_transitions()     ← align all auxiliary arrays (cube state, obstacle, etc.)
        │
        ▼
datasets/processed/{single,multi}_cube/processed_<suffix>.zarr
```

## Three Action Spaces

| Flag | State used | Action dim | Description |
|------|-----------|------------|-------------|
| `ee` | `state_ee[:, :3]` (xyz only) | 3 | End-effector position deltas |
| `ee_full` | `state_ee[:, :7]` (xyz + quaternion) | 6 | Position deltas (3) + Euler angle deltas (3) |
| `joints` | `state_joints[:, :5]` (excludes Jaw) | 5 | Joint angle deltas |

The `ee` space is the simplest — 3D position deltas. The `ee_full` space is more complex because orientation deltas require quaternion math (relative rotation → Euler angles). The `joints` space operates directly in joint angle space.

## Key Functions

### `load_and_merge_zarrs(zarr_paths)`

Loads multiple `.zarr` stores and concatenates them into a single dict of arrays. Episode boundaries (`episode_ends`) are shifted by a cumulative offset so they remain globally correct after concatenation. It also detects DAgger episodes by checking if `"dagger"` appears in the file path.

### `select_action_space(action_space, merged)`

Slices the correct columns from the merged data based on the chosen action space:
- `ee` → first 3 columns of `state_ee` (xyz position)
- `ee_full` → all 7 columns of `state_ee` (position + quaternion)
- `joints` → first 5 columns of `state_joints` (excludes Jaw joint)

### `compute_actions_for_episodes(states, episode_ranges, action_fn)`

The core computation. For each episode:
1. Takes states `[start:end]`
2. Keeps only the first `L-1` timesteps (drops the last one)
3. Computes actions as `action_fn(s_t, s_{t+1})` — defaults to simple subtraction
4. Returns a `keep_idx` array for aligning other data arrays to the same timesteps

For `ee_full`, a custom `action_fn` (`_ee_full_delta`) is used instead of simple subtraction, because orientation deltas require quaternion relative rotation converted to Euler angles.

### `_ee_full_delta(s_curr, s_next)`

Computes 6D actions for the full end-effector space:
1. **Position delta**: simple subtraction `s_next[:3] - s_curr[:3]`
2. **Orientation delta**: `q_relative = q_next * conjugate(q_curr)`, then convert to Euler angles (roll, pitch, yaw)

This gives a 6D action vector: `[dx, dy, dz, droll, dpitch, dyaw]`.

### `trim_to_transitions(merged, keep_idx, skip_keys)`

After computing actions, all auxiliary arrays (cube state, obstacle position, gripper state, etc.) need to be trimmed to match the same timesteps. This function applies `keep_idx` to every array in the merged data, and also handles renaming (`state_ee` → `state_ee_full` to avoid key collisions).

## Output Format

The output `.zarr` store has this structure:

```
processed_<suffix>.zarr/
├── data/
│   ├── state_<suffix>     # (N, D_state) — states for the chosen action space
│   ├── action_<suffix>    # (N, D_action) — computed delta actions
│   ├── action_gripper     # (N, 1) — raw gripper control commands
│   ├── state_gripper      # (N, 1) — gripper state
│   ├── state_cube         # (N, 7) — cube position + orientation
│   ├── state_obstacle     # (N, 3) — obstacle position
│   └── ...                # other auxiliary arrays
├── meta/
│   └── episode_ends       # cumulative end indices per episode
└── attrs: action_space, state_key, action_key, num_episodes, ...
```

## Usage Examples

```bash
# Simplest action space (recommended to start with)
python scripts/compute_actions.py --action-space ee

# Full end-effector control with orientation
python scripts/compute_actions.py --action-space ee_full

# Joint-space control
python scripts/compute_actions.py --action-space joints

# For multicube exercise
python scripts/compute_actions.py --action-space ee --datasets-dir ./datasets/raw/multi_cube

# Custom output path
python scripts/compute_actions.py --action-space ee --output ./my_data.zarr
```
