# Quaternions — The Theory

## The problem with representing 3D rotation

We need a way to describe "how is this object oriented in 3D space?" There are several options:

| Representation | Size | Problem |
|---|---|---|
| Euler angles (roll, pitch, yaw) | 3 numbers | **Gimbal lock** — lose a degree of freedom at certain angles |
| Rotation matrix | 9 numbers (3x3) | Redundant (only 3 DOF), can drift from being a valid rotation |
| Quaternion | 4 numbers | Compact, no gimbal lock, easy to compose |

## What is a quaternion?

A quaternion extends complex numbers. A complex number is $a + bi$ (2D rotation). A quaternion has **three** imaginary units:

$$q = w + xi + yj + zk$$

where $i, j, k$ satisfy:

$$i^2 = j^2 = k^2 = ijk = -1$$

$$ij = k, \quad jk = i, \quad ki = j$$

$$ji = -k, \quad kj = -i, \quad ik = -j$$

In code, stored as `[w, x, y, z]` (MuJoCo convention).

## How quaternions encode rotation

A **unit quaternion** ($\|q\| = w^2 + x^2 + y^2 + z^2 = 1$) represents a rotation of angle $\theta$ around axis $\mathbf{u} = [u_x, u_y, u_z]$:

$$q = \cos\frac{\theta}{2} + \sin\frac{\theta}{2}(u_x i + u_y j + u_z k)$$

So:

- $w = \cos(\theta/2)$ — encodes how much rotation
- $[x, y, z] = \sin(\theta/2) \cdot \mathbf{u}$ — encodes the rotation axis

**Examples:**

| Rotation | $\theta$ | Axis $\mathbf{u}$ | Quaternion $[w, x, y, z]$ |
|---|---|---|---|
| No rotation | $0$ | any | $[1, 0, 0, 0]$ |
| 90 deg around Z | $\pi/2$ | $[0,0,1]$ | $[0.707, 0, 0, 0.707]$ |
| 180 deg around Y | $\pi$ | $[0,1,0]$ | $[0, 0, 1, 0]$ |
| 180 deg around X | $\pi$ | $[1,0,0]$ | $[0, 1, 0, 0]$ |

## Why $\theta/2$? (the deep reason)

This is what makes the math work. To rotate a 3D vector $\mathbf{v}$ by quaternion $q$, you do:

$$\mathbf{v}' = q \cdot \mathbf{v} \cdot q^*$$

where $\mathbf{v}$ is treated as a "pure" quaternion $[0, v_x, v_y, v_z]$ and $q^*$ is the conjugate. The $\theta/2$ ensures that applying $q$ from the left and $q^*$ from the right produces a total rotation of $\theta$ (not $2\theta$).

## The four operations in `utils.py`

### 1. `quat_mul(q1, q2)` — Multiplication

Combines two rotations. If $q_1$ rotates A to B and $q_2$ rotates B to C, then $q_1 \cdot q_2$ rotates A to C.

The multiplication rule (expanding $(w_1 + x_1 i + y_1 j + z_1 k)(w_2 + x_2 i + y_2 j + z_2 k)$ using $ij=k$ etc.):

$$w = w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2$$

$$x = w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2$$

$$y = w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2$$

$$z = w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2$$

**Note:** Order matters! $q_1 \cdot q_2 \neq q_2 \cdot q_1$ (just like 3D rotations are not commutative — rotating pitch-then-yaw gives a different result than yaw-then-pitch).

### 2. `quat_conjugate(q)` — Inverse rotation

$$q^* = [w, -x, -y, -z]$$

For unit quaternions, the conjugate equals the inverse: $q \cdot q^* = [1, 0, 0, 0]$ (no rotation).

If $q$ rotates world to hand, then $q^*$ rotates hand to world.

### 3. `quat_normalize(q)` — Keep it valid

After multiplications, floating point errors accumulate. The quaternion might become $\|q\| = 1.0000003$ instead of exactly 1. Normalization fixes this:

$$q_{norm} = \frac{q}{\|q\|} = \frac{[w,x,y,z]}{\sqrt{w^2+x^2+y^2+z^2}}$$

A non-unit quaternion does not represent a pure rotation (it would also scale things).

### 4. `rot_mat_to_quat(mat)` — Matrix to quaternion

A 3x3 rotation matrix and a unit quaternion represent the same thing — just different formats. MuJoCo stores body orientations as matrices internally, but quaternions are more compact for the observation vector (4 numbers vs 9).

## Comparison to 2D

In 2D, rotation by $\theta$ is just a complex number $e^{i\theta} = \cos\theta + i\sin\theta$. Quaternions are the 3D generalization:

| 2D | 3D |
|---|---|
| Complex number $a + bi$ | Quaternion $w + xi + yj + zk$ |
| 1 imaginary unit $i$ | 3 imaginary units $i, j, k$ |
| Unit circle $\|z\| = 1$ | Unit 4D sphere $\|q\| = 1$ |
| Multiply to compose | Multiply to compose |
| Conjugate = inverse | Conjugate = inverse |
| No gimbal lock (only 1 axis) | No gimbal lock (any axis) |

## How this applies to `get_obs` in Ex3

We need to convert world-frame states into the robot's base frame:

### Positions (world to base frame)

```
offset_w = pos_w - base_pos_w          # offset in world coordinates
pos_base = base_rot_w.T @ offset_w     # rotate into base's local frame
```

`base_rot_w` maps base to world, so its transpose (`base_rot_w.T`) maps world to base.

### Rotations (world to base frame)

```python
ee_quat_w   = rot_mat_to_quat(ee_rot_w)      # EE rotation in world (as quaternion)
base_quat_w = rot_mat_to_quat(base_rot_w)     # Base rotation in world
base_quat_inv = quat_conjugate(base_quat_w)   # Inverse: world -> base
ee_quat_base = quat_mul(base_quat_inv, ee_quat_w)  # EE rotation relative to base
ee_quat_base = quat_normalize(ee_quat_base)   # Clean up floating point drift
```

### Analogy

Think of giving directions:

- **World frame**: "The coffee shop is at 47.3 N, 8.5 E" — only useful if you have a GPS
- **Base frame**: "The coffee shop is 200m ahead and 50m to your left" — works regardless of where you are standing or which way you are facing

The policy is like a person following directions — it works much better with relative instructions.

---

## How Quaternions Were Derived

### The 2D starting point: complex numbers

In 2D, rotation by angle $\theta$ is multiplication by $e^{i\theta} = \cos\theta + i\sin\theta$.

Rotating the point $(x, y)$ by $\theta$:

$$(x + yi)(\cos\theta + i\sin\theta) = \text{rotated point}$$

This works because $i^2 = -1$ makes the algebra produce exactly the rotation matrix equations:

$$x' = x\cos\theta - y\sin\theta$$

$$y' = x\sin\theta + y\cos\theta$$

### Hamilton's question (1843)

William Rowan Hamilton asked: **can we extend complex numbers to 3D?** He wanted a number system where multiplication = 3D rotation, just like complex multiplication = 2D rotation.

He tried **triplets** $a + bi + cj$ for years. The problem: he couldn't define multiplication that preserved lengths ($\|q_1 \cdot q_2\| = \|q_1\| \cdot \|q_2\|$). In 3D with three components, there's no consistent way to define $ij$ without breaking something.

### The breakthrough: you need 4 dimensions, not 3

On October 16, 1843, walking along Brougham Bridge in Dublin, Hamilton realized: **you need four components, not three**. He famously carved the equations into the bridge:

$$i^2 = j^2 = k^2 = ijk = -1$$

From $ijk = -1$ you can derive all the multiplication rules:

Multiply both sides of $ijk = -1$ by $k$ on the right:

$$ijk^2 = -k \implies ij(-1) = -k \implies ij = k$$

Multiply $ijk = -1$ by $i$ on the left:

$$i^2 jk = -i \implies (-1)jk = -i \implies jk = i$$

And so on for all combinations.

### Why does this encode 3D rotation?

The key insight: $i$, $j$, $k$ correspond to 90-degree rotations in the three coordinate planes:

- $i$ = 90 deg rotation in the YZ plane
- $j$ = 90 deg rotation in the XZ plane
- $k$ = 90 deg rotation in the XY plane

And the rule $ij = k$ means: "rotate 90 deg in YZ, then 90 deg in XZ, and you get 90 deg in XY." This is exactly how 3D rotations compose.

### Why the half-angle?

To rotate a vector $\mathbf{v}$ using quaternion multiplication, you need to apply $q$ from both sides:

$$\mathbf{v}' = q \mathbf{v} q^*$$

Each side contributes $\theta/2$ of rotation, giving $\theta$ total. This **sandwich product** is necessary because single-sided multiplication ($q\mathbf{v}$) would also add an unwanted scaling/reflection. The two-sided version cancels that out, leaving only pure rotation.

You can verify: if $q = \cos\frac{\theta}{2} + \sin\frac{\theta}{2} \cdot k$ (rotation around Z), then:

$$q \cdot (xi + yj) \cdot q^* = (x\cos\theta - y\sin\theta)i + (x\sin\theta + y\cos\theta)j$$

Which is exactly the 2D rotation formula — but derived purely from quaternion algebra.

### Why not just use rotation matrices?

You can! But quaternions have practical advantages:

| | Rotation Matrix | Quaternion |
|---|---|---|
| Storage | 9 numbers | 4 numbers |
| Compose two rotations | 27 multiplications | 16 multiplications |
| Interpolation (SLERP) | Hard, can break | Natural and smooth |
| Numerical drift | Matrix stops being orthogonal | Just renormalize to length 1 |

### The mathematical structure

For the mathematically curious: unit quaternions form the group $SU(2)$, which is a **double cover** of the 3D rotation group $SO(3)$. This means:

- Every 3D rotation maps to **two** quaternions: $q$ and $-q$ (same rotation, opposite signs)
- The quaternion space is a 4D unit sphere ($S^3$), which is topologically simpler than $SO(3)$
- This is why interpolation and optimization work better with quaternions — the space is smoother

Hamilton sacrificed commutativity ($q_1 q_2 \neq q_2 q_1$) to get everything else. This is the minimum price: it has been proven (Frobenius theorem, 1878) that the only division algebras over the reals are $\mathbb{R}$ (1D), $\mathbb{C}$ (2D), $\mathbb{H}$ quaternions (4D), and $\mathbb{O}$ octonions (8D). There is no 3D version — Hamilton was right to go to 4.

## What is gimbal lock?

### The setup

Euler angles represent rotation as three sequential rotations: roll, pitch, yaw (or X, Y, Z — the order varies).

Imagine three nested rings (gimbals), each rotating around one axis:

```
Outer ring:    rotates around Z (yaw)
  Middle ring:   rotates around Y (pitch)
    Inner ring:    rotates around X (roll)
      Object inside
```

### What goes wrong

**Gimbal lock** happens when the middle rotation (pitch) reaches 90 degrees. At that point, the outer and inner rings end up rotating around the **same axis** — you lose one degree of freedom.

```
Before (3 independent axes):        After pitch = 90 deg (gimbal lock):

   Z  Y                                Z
   | /                                 |
   |/___  X                       X ---+--- Y   X and Z now do the same thing!
                                       |
```

Now yaw and roll both rotate around the same direction. You can't distinguish between them — you've lost the ability to rotate in one direction.

### Real-world consequence

In a flight simulator using Euler angles, if the plane points straight up, the controls for "turn left" and "spin" become identical. You can't smoothly pass through that orientation.

### Why quaternions fix this

Quaternions don't decompose rotation into sequential steps around fixed axes. They encode rotation as a single axis + angle, so no axis can ever "collapse" into another. Every orientation has a smooth, unique path to every other orientation.

For this homework, gimbal lock is mostly background knowledge. The practical takeaway: the observation uses quaternions (4 numbers) instead of Euler angles (3 numbers) because quaternions always work, no edge cases.

## What does `w` mean in a quaternion?

Quaternions represent **rotation, not position**. The `[w, x, y, z]` have nothing to do with spatial coordinates.

To describe a 3D rotation, you need two things:

1. **Which axis** to rotate around (a direction vector, needs 3 numbers)
2. **How much** to rotate (an angle, needs 1 number)

A quaternion packs both into 4 numbers:

```
q = [w,    x,    y,    z   ]
     |     |_____|_____|
     |           |
     |           sin(theta/2) * axis direction [ux, uy, uz]
     |           (WHICH WAY to rotate)
     |
     cos(theta/2)
     (HOW MUCH to rotate)
```

**So `w` encodes the amount of rotation:**

- `w = 1` means $\theta = 0$ means no rotation at all
- `w = 0.707` means $\theta = 90$ deg
- `w = 0` means $\theta = 180$ deg (maximum rotation)

**Position** (where something is) is just a regular 3D vector `[x, y, z]` — no quaternion needed. Quaternions only describe **orientation** (which way something is facing).

| Concept | Representation | Example |
|---|---|---|
| **Position** (where) | `[x, y, z]` — 3D vector | "the hand is at [0.3, 0.1, 0.25]" |
| **Orientation** (which way) | `[w, x, y, z]` — quaternion | "the hand is rotated 90 deg around Z" |

In the `get_obs` function, the observation includes both:

- `ee_pos_base` (3D vector) — **where** the hand is
- `ee_quat_base` (quaternion) — **which way** the hand is facing
