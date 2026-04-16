# Exercise 1: MDP - Policy Iteration & Value Iteration

## Overview

This file implements two classic **Dynamic Programming** algorithms for solving finite tabular MDPs (Markov Decision Processes): **Policy Iteration** and **Value Iteration**.

Both operate on an environment with:
- `env.n_states` — number of states
- `env.n_actions` — number of actions
- `env.P[s][a]` — transition model: a list of `(prob, next_state, reward, done)` tuples

---

## Core Concept: Bellman Equations

Both algorithms revolve around the Bellman equation. For a policy `pi`, the value of state `s` is:

```
V(s) = sum_a pi(s,a) * sum_{s',r} P(s'|s,a) * [r + gamma * V(s')]
```

The optimal value satisfies:

```
V*(s) = max_a sum_{s',r} P(s'|s,a) * [r + gamma * V*(s')]
```

---

## Class 1: `PolicyIteration`

Policy Iteration alternates between two steps until the policy stabilizes.

### Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `v` | `(n_states,)` | State-value function, initialized to zeros |
| `pi` | `(n_states, n_actions)` | Stochastic policy (probability distribution per state), initialized to uniform random |
| `theta` | scalar | Convergence threshold (default `1e-3`) |
| `gamma` | scalar | Discount factor (default `0.9`) |

### Method: `policy_evaluation()`

**Goal**: Compute `V^pi` — the value function for the current policy `self.pi`.

**Algorithm** (iterative):
1. For each state `s`, compute the new value:
   ```
   new_v[s] = sum_a pi[s][a] * Q(s, a)
   ```
   where:
   ```
   Q(s, a) = sum over (prob, s', r, done) in P[s][a]:
                prob * (r + gamma * V(s'))
   ```
2. Track `max_diff = max |new_v[s] - old_v[s]|` across all states.
3. Repeat until `max_diff < theta` (convergence).

**Key detail**: `qsa_list` collects `pi[s][a] * Q(s,a)` for each action, then `new_v[s] = sum(qsa_list)`.

### Method: `policy_improvement()`

**Goal**: Make the policy greedy w.r.t. the current value function.

**Algorithm**:
1. For each state `s`, compute `Q(s, a)` for all actions `a`.
2. Find the maximum Q-value: `max_q = max(qsa_list)`.
3. Assign equal probability to all actions achieving `max_q` (tie-breaking), zero to others.

**Returns**: The updated policy `self.pi`.

### Method: `policy_iteration()`

**Goal**: Run the full algorithm until convergence.

**Algorithm**:
```
repeat:
    old_pi = copy(pi)
    policy_evaluation()      # step 1: evaluate current policy
    policy_improvement()     # step 2: improve policy greedily
    if pi == old_pi: break   # step 3: stop if policy unchanged
return v, pi
```

---

## Class 2: `ValueIteration`

Value Iteration directly computes the optimal value function, then extracts the policy at the end.

### Attributes

Same as `PolicyIteration`, except `pi` is initialized to zeros (not uniform) since it's only set at the end.

### Method: `value_iteration()`

**Goal**: Compute `V*` — the optimal value function.

**Algorithm** (iterative):
1. For each state `s`, compute:
   ```
   new_v[s] = max_a Q(s, a)
   ```
   where `Q(s, a)` is the same Bellman backup as before:
   ```
   Q(s, a) = sum over (prob, s', r, done) in P[s][a]:
                prob * (r + gamma * V(s'))
   ```
2. Track `max_diff` and repeat until `max_diff < theta`.
3. After convergence, call `get_policy()` to extract the greedy policy.

**Key difference from Policy Iteration**: Instead of `sum_a pi[s][a] * Q(s,a)`, it takes `max_a Q(s,a)`. This combines evaluation and improvement into a single step.

### Method: `get_policy()`

**Goal**: Extract the greedy policy from the converged value function.

Identical logic to `policy_improvement()` in PolicyIteration — compute Q-values per state, assign equal probability to all maximizing actions.

---

## Comparison

| | Policy Iteration | Value Iteration |
|---|---|---|
| **Per iteration** | Full policy eval (inner loop) + one improvement | Single Bellman optimality backup |
| **Convergence** | Fewer outer iterations | More iterations but each is cheaper |
| **Policy update** | After full evaluation | Extracted once at the end |
| **Value update rule** | `sum_a pi(s,a) * Q(s,a)` | `max_a Q(s,a)` |

---

## What the TODOs Ask You to Implement

1. **Q-value computation** (appears 4 times): Loop over `self.env.P[s][a]` transitions, accumulate `prob * (r + gamma * V(s'))` into `qsa`.
2. **Convergence check** (appears 2 times): Break the while loop when `max_diff < self.theta`.
3. **Policy iteration main loop**: Call `policy_evaluation()` then `policy_improvement()`, store result as `new_pi`.

The Q-value computation is the same pattern everywhere — only the outer aggregation differs (`sum` with policy weights vs. `max`).
