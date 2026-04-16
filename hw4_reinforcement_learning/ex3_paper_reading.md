# Exercise 3: Paper Reading Guide

## Paper 1: Proximal Policy Optimization Algorithms (Schulman et al., 2017)

### Background

Before PPO, the main approaches to policy gradient were:
- **Vanilla Policy Gradient (REINFORCE)**: Simple but high variance, sensitive to step size
- **TRPO**: Constrains KL divergence between old and new policy, but requires second-order optimization, making it complex and hard to scale

PPO asks: can we get TRPO-like stability with first-order (Adam) optimization?

### Core Problem

Policy gradient methods are fragile. Too large a step -- catastrophically bad policy. Too small -- painfully slow. TRPO solves this with a hard KL constraint but the implementation is complex.

### Key Ideas

**1. The Surrogate Objective**

Standard policy gradient: `L = E[ log pi(a|s) * A(s,a) ]`

TRPO introduced importance-sampled surrogate:
```
L = E[ r(theta) * A(s,a) ]    where r(theta) = pi_new(a|s) / pi_old(a|s)
```
Problem: without a constraint, maximizing this can produce huge ratios (r >> 1), leading to destructive updates.

**2. Clipped Surrogate Objective (PPO's main contribution)**

```
L_CLIP = E[ min( r * A, clip(r, 1-eps, 1+eps) * A ) ]
```
where eps is typically 0.2.

How it works:
- A > 0 (good action): r can increase but is capped at 1+eps. Can't over-commit.
- A < 0 (bad action): r can decrease but is floored at 1-eps. Can't swing too far.
- The `min` always takes the more pessimistic (conservative) estimate.

**3. Multiple Epochs**

PPO reuses the same batch for multiple gradient steps (3-10 epochs). Clipping prevents the policy from drifting too far during repeated updates. More sample-efficient than vanilla PG.

### Algorithm

```
for iteration = 1, 2, ...
    Collect T timesteps with current policy
    Compute advantages (GAE)
    for epoch = 1, ..., K:
        for each mini-batch:
            r = pi_new / pi_old
            Compute L_CLIP, L_VF, entropy
            Update theta with Adam
```

### Connection to Homework

The total loss in practice (and in your code):
```
L = L_CLIP - c1 * L_VF + c2 * entropy
```
This is exactly what your `update()` computes: `surrogate_loss + value_loss + entropy_loss`.

---

## Paper 2: High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)

### Core Problem

Advantage estimation has a fundamental tradeoff:
- **Monte Carlo** (sum all future rewards): unbiased but high variance
- **TD(0)** (r + gamma*V(s') - V(s)): low variance but biased (depends on V accuracy)

GAE provides a smooth interpolation.

### Key Ideas

**1. n-step Advantage Estimators**

TD residual: `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`

```
A^(1) = delta_t                                     (1-step, low var, high bias)
A^(2) = delta_t + gamma * delta_{t+1}               (2-step)
...
A^(inf) = sum gamma^l * delta_{t+l}                  (Monte Carlo, no bias, high var)
```

**2. GAE: Exponentially-Weighted Average**

```
A_t^GAE = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
```

**3. Lambda controls the tradeoff**

- **lambda = 0**: `A = delta_t` -- pure 1-step TD. Low variance, high bias.
- **lambda = 1**: Monte Carlo return minus V(s). Unbiased, high variance.
- **lambda = 0.95** (your homework): Sweet spot for most continuous control.

**4. Efficient recursive computation (backwards through trajectory)**

```
A_T = 0
for t = T-1, ..., 0:
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t = delta_t + gamma * lambda * A_{t+1}
```

### Connection to Homework

Config: `gamma=0.99, gae_lambda=0.95`. The `RolloutBuffer` computes GAE internally. You use the resulting `adv_batch` in `compute_surrogate_loss`.

---

## How Both Papers Connect

```
Collect rollout --> GAE advantages (Paper 2) --> PPO clipped update (Paper 1)
```

- **GAE** controls the quality of the gradient signal (bias-variance of advantages)
- **PPO** controls the size of the policy update (clipping prevents destructive steps)

Both are needed: good advantages with reckless updates = instability. Bad advantages with careful updates = slow/wrong learning.
