# `get_obs` — Building the Observation Vector

## What is `get_obs`?

The RL policy needs to "see" the world to make decisions. `get_obs` assembles a flat numpy array of numbers that describes the current state — this is what the neural network receives as input.

## The inputs explained

All positions and rotations from MuJoCo come in the **world frame** (absolute coordinates). We need to convert them to the **base frame** (relative to the robot).

### Concrete example setup

Imagine the robot base is at position `[1, 0, 0]` in the world and rotated 90 degrees around Z:

```
World frame:                    Base frame (robot's perspective):

    Y                               Y_base (= world -X)
    |                                |
    |   base at [1,0,0]             base
    |   facing Y direction           |
    +-------> X                      +-------> X_base (= world Y)
```

Now let's say:

```python
qpos        = [0.1, -0.5, 1.2, 0.0, 0.3, 0.0]   # joint angles (6 joints)
ee_pos_w    = [0.8, 0.3, 0.25]                     # EE position in world
ee_rot_w    = np.array([[0, -1, 0],                 # EE rotation in world (3x3 matrix)
                        [1,  0, 0],
                        [0,  0, 1]])
base_pos_w  = [1.0, 0.0, 0.0]                      # base position in world
base_rot_w  = np.array([[0, -1, 0],                 # base rotation in world (90 deg around Z)
                        [1,  0, 0],
                        [0,  0, 1]])
target_pos_w = [1.2, 0.1, 0.3]                     # target position in world
```

## Step-by-step transformation

### 1. `qpos` — joint positions (no conversion needed)

Joint angles are already relative to the robot. They don't depend on where the robot is in the world.

```python
qpos = [0.1, -0.5, 1.2, 0.0, 0.3, 0.0]  # shape: (6,)
```

### 2. `ee_pos_base` — end-effector position in base frame

**Step 2a:** Subtract base position to get offset in world frame:

```python
offset_w = ee_pos_w - base_pos_w
         = [0.8, 0.3, 0.25] - [1.0, 0.0, 0.0]
         = [-0.2, 0.3, 0.25]
```

This says: "the EE is 0.2m behind, 0.3m right, 0.25m up from the base" — **in world coordinates**.

**Step 2b:** Rotate into the base's local frame:

```python
ee_pos_base = base_rot_w.T @ offset_w
```

Why `.T` (transpose)? `base_rot_w` maps base-frame vectors to world-frame vectors. We want the reverse (world to base), so we use the transpose (which equals the inverse for rotation matrices).

```python
base_rot_w.T = [[0,  1, 0],       # transpose of the 90-deg-Z rotation
                [-1, 0, 0],
                [0,  0, 1]]

ee_pos_base = base_rot_w.T @ [-0.2, 0.3, 0.25]
            = [0*(-0.2) + 1*(0.3) + 0*(0.25),
               -1*(-0.2) + 0*(0.3) + 0*(0.25),
               0*(-0.2) + 0*(0.3) + 1*(0.25)]
            = [0.3, 0.2, 0.25]
```

From the robot's perspective: "the EE is 0.3m forward, 0.2m to the right, 0.25m up."

Shape: `(3,)`

### 3. `ee_quat_base` — end-effector orientation in base frame

**Step 3a:** Convert rotation matrices to quaternions:

```python
ee_quat_w   = rot_mat_to_quat(ee_rot_w)    # e.g. [0.707, 0, 0, 0.707]
base_quat_w = rot_mat_to_quat(base_rot_w)  # e.g. [0.707, 0, 0, 0.707]
```

**Step 3b:** Get the inverse of the base rotation:

```python
base_quat_inv = quat_conjugate(base_quat_w)  # [0.707, 0, 0, -0.707]
```

This "undoes" the base rotation.

**Step 3c:** Compose to get relative rotation:

```python
ee_quat_base = quat_mul(base_quat_inv, ee_quat_w)
```

The logic: "undo the base rotation, then apply the EE rotation" = "what's the EE rotation relative to the base?"

In this example, since the base and EE have the same rotation in the world frame:

```python
ee_quat_base = quat_mul(base_quat_inv, ee_quat_w)
             = [1, 0, 0, 0]   # identity = no relative rotation
```

This makes sense — the EE is facing the same direction as the base.

**Step 3d:** Normalize to prevent floating point drift:

```python
ee_quat_base = quat_normalize(ee_quat_base)  # ensure ||q|| = 1
```

Shape: `(4,)`

### 4. `target_pos_base` — target position in base frame

Same logic as the EE position:

```python
offset_w = target_pos_w - base_pos_w
         = [1.2, 0.1, 0.3] - [1.0, 0.0, 0.0]
         = [0.2, 0.1, 0.3]

target_pos_base = base_rot_w.T @ offset_w
                = [0.1, -0.2, 0.3]
```

From the robot's perspective: "the target is 0.1m forward, 0.2m to the left, 0.3m up."

Shape: `(3,)`

## Final observation vector

Concatenate everything into one flat array:

```python
obs = np.concatenate([
    qpos,            # (6,)  joint angles
    ee_pos_base,     # (3,)  EE position relative to base
    ee_quat_base,    # (4,)  EE orientation relative to base
    target_pos_base, # (3,)  target position relative to base
])
# Total shape: (16,)
```

## Summary diagram

```
World frame inputs                    Base frame outputs
─────────────────                     ──────────────────

qpos ─────────────────────────────────> qpos (unchanged)

ee_pos_w ──┐                    ┌────> ee_pos_base (3,)
            ├─ subtract ─ rotate ┤
base_pos_w ─┘   base_pos  base_rot.T

ee_rot_w ──┐                    ┌────> ee_quat_base (4,)
            ├─ to_quat ─ quat_mul ┤
base_rot_w ─┘   conjugate(base)

target_pos_w ─┐                 ┌────> target_pos_base (3,)
               ├─ subtract ─ rotate ┤
base_pos_w ────┘   base_pos  base_rot.T

                                       ──────────────────
                                       Concatenate → obs (16,)
```

## Why not just use world frame?

If you train the policy with the robot at `[0, 0, 0]` facing X, it learns things like "move joints so EE goes to [0.3, 0.1, 0.25]." If you then place the robot at `[5, 3, 0]` facing Y, those absolute coordinates are meaningless.

By converting to the base frame, the policy learns "move joints so EE goes 0.3m forward" — which works regardless of where the robot is placed.
