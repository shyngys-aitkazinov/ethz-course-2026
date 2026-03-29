# Inverse Kinematics (`ik_track`) — Detailed Explanation

## The Problem

You have a robot arm with 6 joints. Each joint has an angle $q_i$. Together they form a vector:

$$\mathbf{q} = [q_1, q_2, q_3, q_4, q_5, q_6]^T$$

**Forward Kinematics (FK)** is the function that maps joint angles to end-effector position:

$$\mathbf{x}_{ee} = f(\mathbf{q}) \in \mathbb{R}^3$$

For example: $f([0, -1, 1.5, 0.3, 0, 0]) = [0.28, 0.03, 0.18]$

**Inverse Kinematics (IK)** is the reverse: given a desired position $\mathbf{x}_{target}$, find joint angles $\mathbf{q}^*$ such that:

$$f(\mathbf{q}^*) = \mathbf{x}_{target}$$

This is hard because $f$ is nonlinear and may have multiple solutions (or none). We solve it iteratively.

---

## The Arguments

| Argument | Type | What it is | Example |
|---|---|---|---|
| `model` | `MjModel` | Robot blueprint — link lengths, joint limits, masses | Loaded from XML |
| `data` | `MjData` | Robot state — `qpos`, `qvel`, site positions | Changes each step |
| `site_name` | `str` | The point we want to control | `"ee_site"` (fingertip) |
| `target_pos` | `np.ndarray(3,)` | Where we want the point to go | `[0.3, 0.1, 0.25]` |
| `damping` | `float` | Regularization parameter $\lambda$ | `1e-3` |
| `pos_gain` | `float` | Proportional gain $K_p$ | `2.0` |
| `dt` | `float` | Integration step size $\Delta t$ | `0.1` |
| `max_iters` | `int` | Safety limit on iterations | `2000` |

---

## The Algorithm Step by Step

### Step 1: Compute the Position Error

At each iteration, we compute where the end-effector currently is and how far off it is:

$$\mathbf{e} = \mathbf{x}_{target} - \mathbf{x}_{ee}$$

**Concrete example:**
- Target: $\mathbf{x}_{target} = [0.30, 0.10, 0.25]$
- Current EE: $\mathbf{x}_{ee} = [0.22, 0.07, 0.19]$
- Error: $\mathbf{e} = [0.08, 0.03, 0.06]$
- Error magnitude: $\|\mathbf{e}\| = \sqrt{0.08^2 + 0.03^2 + 0.06^2} = 0.104$

If $\|\mathbf{e}\| < 10^{-3}$ (less than 1mm), we stop — close enough.

```python
ee_pos = data.site(site_name).xpos.copy()
err_pos = target_pos - ee_pos
if np.linalg.norm(err_pos) < 1e-3:
    break
```

### Step 2: Compute the Jacobian

The **Jacobian** $J$ is the derivative of the FK function. It tells us how small changes in joint angles affect the end-effector position:

$$J = \frac{\partial f(\mathbf{q})}{\partial \mathbf{q}} \in \mathbb{R}^{3 \times 6}$$

Each column of $J$ corresponds to one joint. Each row corresponds to one spatial direction (x, y, z):

$$J = \begin{bmatrix} \frac{\partial x_{ee}}{\partial q_1} & \frac{\partial x_{ee}}{\partial q_2} & \cdots & \frac{\partial x_{ee}}{\partial q_6} \\ \frac{\partial y_{ee}}{\partial q_1} & \frac{\partial y_{ee}}{\partial q_2} & \cdots & \frac{\partial y_{ee}}{\partial q_6} \\ \frac{\partial z_{ee}}{\partial q_1} & \frac{\partial z_{ee}}{\partial q_2} & \cdots & \frac{\partial z_{ee}}{\partial q_6} \end{bmatrix}$$

**Reading the Jacobian — example:**

| | Rotation | Pitch | Elbow | Wrist_Pitch | Wrist_Roll | Jaw |
|---|---|---|---|---|---|---|
| $\partial x_{ee}$ | +0.00 | -0.28 | -0.15 | -0.06 | +0.00 | +0.00 |
| $\partial y_{ee}$ | +0.28 | +0.00 | +0.00 | +0.00 | -0.06 | +0.00 |
| $\partial z_{ee}$ | +0.00 | +0.05 | +0.12 | +0.06 | +0.00 | +0.00 |

This tells us:
- Rotating the **Pitch** joint by 1 rad moves the EE by $-0.28$ in $x$ and $+0.05$ in $z$
- The **Jaw** joint barely affects EE position (it opens/closes the gripper)
- The **Rotation** joint mainly moves the EE in $y$ (swinging left/right)

In the code, MuJoCo computes both position and rotation Jacobians, stacked into a 6x6 matrix:

$$J_{full} = \begin{bmatrix} J_{pos} \\ J_{rot} \end{bmatrix} \in \mathbb{R}^{6 \times 6}$$

```python
jacp = np.zeros((3, num_joints))  # position Jacobian
jacr = np.zeros((3, num_joints))  # rotation Jacobian
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
J = np.vstack([jacp, jacr])  # shape (6, 6)
```

### Step 3: Damped Least Squares (DLS)

#### The naive approach (and why it fails)

If $J$ were square and invertible, we could simply do:

$$\dot{\mathbf{q}} = J^{-1} \mathbf{e}$$

But this fails near **singularities** — configurations where the robot loses a degree of freedom (e.g., arm fully stretched). Near singularities, $J$ becomes nearly rank-deficient, and $J^{-1}$ produces huge values.

#### The pseudoinverse approach

For non-square or rank-deficient $J$, use the Moore-Penrose pseudoinverse:

$$\dot{\mathbf{q}} = J^T (J J^T)^{-1} \mathbf{e}$$

This minimizes $\|\dot{\mathbf{q}}\|$ while satisfying $J \dot{\mathbf{q}} = \mathbf{e}$. But it still blows up near singularities because $(J J^T)^{-1}$ becomes huge.

#### The DLS solution (what we use)

Add a damping term $\lambda^2 I$ to regularize:

$$\dot{\mathbf{q}} = J^T (J J^T + \lambda^2 I)^{-1} \mathbf{e}$$

The matrix $J J^T + \lambda^2 I$ is always invertible (positive definite), so the solution is always bounded.

**What $\lambda$ does intuitively:**
- $\lambda \to 0$: behaves like the pseudoinverse (fast but unstable near singularities)
- $\lambda$ large: very stable but slow convergence (the robot barely moves each step)
- $\lambda = 10^{-3}$: a good balance for this robot

#### The math problem DLS solves

DLS solves a **regularized least-squares** (Tikhonov regularization) problem:

$$\min_{\dot{\mathbf{q}}} \left( \|J\dot{\mathbf{q}} - \mathbf{e}\|^2 + \lambda^2 \|\dot{\mathbf{q}}\|^2 \right)$$

Two competing objectives:

| Term | Meaning |
|---|---|
| $\|J\dot{\mathbf{q}} - \mathbf{e}\|^2$ | Move the end-effector as close to the target as possible |
| $\lambda^2 \|\dot{\mathbf{q}}\|^2$ | But don't use crazy-large joint velocities to do it |

Taking the derivative w.r.t. $\dot{\mathbf{q}}$ and setting it to zero:

$$\frac{\partial}{\partial \dot{\mathbf{q}}} \left( \|J\dot{\mathbf{q}} - \mathbf{e}\|^2 + \lambda^2 \|\dot{\mathbf{q}}\|^2 \right) = 2J^T(J\dot{\mathbf{q}} - \mathbf{e}) + 2\lambda^2 \dot{\mathbf{q}} = 0$$

$$J^T J \dot{\mathbf{q}} + \lambda^2 \dot{\mathbf{q}} = J^T \mathbf{e}$$

$$(J^T J + \lambda^2 I) \dot{\mathbf{q}} = J^T \mathbf{e}$$

This gives the **normal form** (also called the **primal form**):

$$(J^T J + \lambda^2 I) \dot{\mathbf{q}} = J^T \mathbf{e} \quad \Rightarrow \quad \dot{\mathbf{q}} = (J^T J + \lambda^2 I)^{-1} J^T \mathbf{e}$$

Here $(J^T J + \lambda^2 I)$ is an $n_v \times n_v$ matrix (6x6 for our robot).

#### Why we use the dual form instead

The **dual form** gives the same answer but works with a smaller matrix when we have more joints than task-space dimensions. Here's the derivation:

**Starting point:** We know the solution lives in the row space of $J$, so we can write:

$$\dot{\mathbf{q}} = J^T \mathbf{z}$$

for some vector $\mathbf{z} \in \mathbb{R}^6$. Substituting into the normal equation:

$$(J^T J + \lambda^2 I) J^T \mathbf{z} = J^T \mathbf{e}$$

Multiply both sides on the left by $J$:

$$J(J^T J + \lambda^2 I) J^T \mathbf{z} = J J^T \mathbf{e}$$

Using the identity $J(J^T J + \lambda^2 I) = (JJ^T + \lambda^2 I)J$ (you can verify by expanding both sides):

$$(JJ^T + \lambda^2 I) J J^T \mathbf{z} = J J^T \mathbf{e}$$

This is satisfied when:

$$(JJ^T + \lambda^2 I) \mathbf{z} = \mathbf{e}$$

So $\mathbf{z} = (JJ^T + \lambda^2 I)^{-1} \mathbf{e}$, and therefore:

$$\dot{\mathbf{q}} = J^T \mathbf{z} = J^T(JJ^T + \lambda^2 I)^{-1}\mathbf{e}$$

This is the **dual form**.

#### Normal vs Dual — when to use which

| Form | Solves | Matrix size | Use when |
|---|---|---|---|
| **Normal** $(J^TJ + \lambda^2 I)^{-1}J^T\mathbf{e}$ | $n_v \times n_v$ system | $6 \times 6$ for our robot | More joints than task dims ($n_v > m$) — but matrix is larger |
| **Dual** $J^T(JJ^T + \lambda^2 I)^{-1}\mathbf{e}$ | $m \times m$ system | $6 \times 6$ for our robot | Fewer task dims than joints ($m < n_v$) — smaller matrix |

In our case $n_v = 6$ joints and $m = 6$ task dimensions (3 pos + 3 rot), so both forms give the same size matrix. But in general, the dual form is preferred in robotics because the task space (typically 6D) is usually smaller than the joint space (could be 7+ for redundant arms).

**In the code**, we use the dual form:

```python
# Solve (JJ^T + λ²I)z = e  for z,  then  qdot = J^T z
A = J @ J.T + damping * np.eye(6)   # 6x6 matrix
sol = np.linalg.solve(A, err_full)   # sol = z
qdot = J.T @ sol                     # qdot = J^T z
```

We use `np.linalg.solve` instead of computing the inverse explicitly because solving $A\mathbf{z} = \mathbf{e}$ is numerically more stable and faster than computing $A^{-1}\mathbf{e}$.

So it's a **trade-off**: accuracy vs. safety. When $\lambda = 0$, you get the exact pseudoinverse (pure accuracy). When $\lambda > 0$, you sacrifice some tracking accuracy to keep joint velocities bounded — critical near singularities where the "exact" solution demands infinite joint speeds.

This is the same math as **Ridge Regression** in machine learning — if you've seen that, it's the same idea applied to robot control.

#### Building the weighted error vector

Since we have a 6x6 Jacobian (position + rotation) but only care about position, we construct a 6D error:

$$\mathbf{e}_{full} = \begin{bmatrix} K_p \cdot \mathbf{e}_{pos} \\ \mathbf{0}_3 \end{bmatrix} \in \mathbb{R}^6$$

where $K_p = 2.0$ is the position gain (how aggressively to correct).

Setting the rotation part to zero means: "I don't care what orientation the hand has, just get it to the right position."

#### Solving the linear system

Instead of computing the inverse $(JJ^T + \lambda^2 I)^{-1}$ directly (numerically unstable), we solve the linear system:

$$(J J^T + \lambda^2 I) \mathbf{x} = \mathbf{e}_{full}$$

for $\mathbf{x}$, then compute:

$$\dot{\mathbf{q}} = J^T \mathbf{x}$$

**Concrete example:**

Let's say $J J^T + \lambda^2 I$ is:

$$A = \begin{bmatrix} 0.085 & 0.002 & -0.01 & 0 & 0 & 0 \\ 0.002 & 0.079 & 0.005 & 0 & 0 & 0 \\ -0.01 & 0.005 & 0.041 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0.001 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0.001 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0.001 \end{bmatrix}$$

And $\mathbf{e}_{full} = [0.16, 0.06, 0.12, 0, 0, 0]^T$ (with $K_p = 2.0$ applied).

Solving $A\mathbf{x} = \mathbf{e}_{full}$ gives some $\mathbf{x}$, then $\dot{\mathbf{q}} = J^T \mathbf{x}$ gives us how much to change each joint.

```python
err_full = np.concatenate([pos_gain * err_pos, np.zeros(3)])
A = J @ J.T + damping * np.eye(6)
sol = np.linalg.solve(A, err_full)
qdot = J.T @ sol
```

### Step 4: Update Joint Angles

Clamp to avoid overshooting, then integrate:

$$\dot{\mathbf{q}} = \text{clip}(\dot{\mathbf{q}}, -2, +2)$$

$$\mathbf{q}_{new} = \mathbf{q}_{old} + \dot{\mathbf{q}} \cdot \Delta t$$

With $\Delta t = 0.1$:

| Joint | $\dot{q}$ (clamped) | $\Delta q = \dot{q} \cdot 0.1$ | $q_{old}$ | $q_{new}$ |
|---|---|---|---|---|
| Rotation | +0.3 | +0.03 | 0.00 | 0.03 |
| Pitch | -1.8 | -0.18 | 0.00 | -0.18 |
| Elbow | +1.2 | +0.12 | 0.00 | 0.12 |
| Wrist_Pitch | +0.5 | +0.05 | 0.00 | 0.05 |
| Wrist_Roll | +0.1 | +0.01 | 0.00 | 0.01 |
| Jaw | +0.0 | +0.00 | 0.00 | 0.00 |

```python
qdot = np.clip(qdot, -2.0, 2.0)
data.qpos[:] += qdot * dt
```

Then we loop back to Step 1 with the updated joint angles.

### Step 5: Convergence

After the loop converges (or hits `max_iters`), we:
1. Save the solution: `target_qpos = data.qpos.copy()`
2. Restore the original joint configuration (because IK is just a *computation* — we didn't actually want to move the robot yet)
3. Return `target_qpos`

---

## Convergence Example (Full Trace)

Target: $[0.30, 0.10, 0.25]$

| Iter | EE Position | $\|\mathbf{e}\|$ | Status |
|---|---|---|---|
| 0 | $[0.15, 0.00, 0.35]$ | 0.198 | Starting... |
| 5 | $[0.22, 0.06, 0.29]$ | 0.097 | Getting closer |
| 15 | $[0.28, 0.09, 0.26]$ | 0.024 | Almost there |
| 30 | $[0.30, 0.10, 0.251]$ | 0.003 | Fine-tuning |
| 42 | $[0.300, 0.100, 0.250]$ | 0.0008 | Converged! |

---

## Why Damped Least Squares? A Visual Intuition

Imagine you're playing darts blindfolded. Someone tells you "you're 10cm left and 5cm high."

| Method | Strategy | Problem |
|---|---|---|
| **Inverse** $J^{-1}\mathbf{e}$ | Move your hand exactly to correct the error in one shot | Near singularity, this says "move your arm 10 meters" |
| **Pseudoinverse** $J^T(JJ^T)^{-1}\mathbf{e}$ | Minimum-effort correction | Still blows up near singularity |
| **DLS** $J^T(JJ^T + \lambda^2 I)^{-1}\mathbf{e}$ | Moderate correction, trading accuracy for stability | Always bounded, converges in more iterations but safely |

The damping $\lambda$ is like saying "I'd rather undershoot than overshoot." With more iterations, we still get there — just more carefully.

---

## Singularity Example

A **singularity** occurs when the robot loses a degree of freedom. For example:

- Arm fully stretched out: can't move the EE further away from the base
- Two joint axes aligned: moving either joint has the same effect

At a singularity, some singular values of $J$ approach zero. The pseudoinverse amplifies these, causing $\|\dot{\mathbf{q}}\| \to \infty$.

With DLS, the singular value $\sigma_i$ gets replaced by:

$$\frac{\sigma_i}{\sigma_i^2 + \lambda^2}$$

When $\sigma_i \to 0$: $\frac{\sigma_i}{\sigma_i^2 + \lambda^2} \to \frac{0}{\lambda^2} = 0$ (safe!)

When $\sigma_i$ is large: $\frac{\sigma_i}{\sigma_i^2 + \lambda^2} \approx \frac{1}{\sigma_i}$ (same as normal inverse)

---

## Summary of Key Equations

| What | Equation |
|---|---|
| Position error | $\mathbf{e} = \mathbf{x}_{target} - f(\mathbf{q})$ |
| Jacobian | $J = \frac{\partial f}{\partial \mathbf{q}}$ |
| DLS solution | $\dot{\mathbf{q}} = J^T(JJ^T + \lambda^2 I)^{-1} K_p \mathbf{e}$ |
| Joint update | $\mathbf{q} \leftarrow \mathbf{q} + \dot{\mathbf{q}} \Delta t$ |
| Convergence check | $\|\mathbf{e}\| < 10^{-3}$ |
