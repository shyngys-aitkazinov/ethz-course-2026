# Homework 4: Theoretical Answers (Full Detail)

---

## Exercise 1: Dynamic Programming

### 1. What is the difference between policy iteration and value iteration in terms of their update procedures?

**Policy Iteration** consists of two strictly separated phases that alternate:

**Phase 1 -- Policy Evaluation:** Given a fixed policy pi, compute the exact value function V^pi by iterating the Bellman expectation equation:
```
V(s) = sum_a pi(s,a) * sum_{s',r} P(s'|s,a) * [r + gamma * V(s')]
```
This is an inner loop that runs until convergence (|V_new - V_old| < theta for all states). The key point is that we compute V^pi **exactly** (up to theta tolerance) before moving on.

**Phase 2 -- Policy Improvement:** Once we have the converged V^pi, we construct a new greedy policy:
```
pi_new(s) = argmax_a sum_{s',r} P(s'|s,a) * [r + gamma * V^pi(s')]
```
We assign equal probability to all actions that achieve the maximum Q-value (tie-breaking).

The outer loop alternates these two phases until the policy stops changing (pi_new == pi_old). The policy improvement theorem guarantees that each improvement step produces a policy that is at least as good as the previous one, so convergence is guaranteed.

**Value Iteration** collapses both phases into a single update. Instead of fully evaluating a policy, it applies the Bellman **optimality** equation directly:
```
V(s) = max_a sum_{s',r} P(s'|s,a) * [r + gamma * V(s')]
```
Notice the `max` instead of `sum_a pi(s,a) * ...`. This is equivalent to doing just one sweep of policy evaluation followed by policy improvement, then immediately moving on. There is no inner convergence loop.

After the value function converges (|V_new - V_old| < theta), the greedy policy is extracted once at the very end.

**Tradeoff:**
- Policy iteration needs fewer outer iterations because each iteration fully solves for V^pi. In practice, it often converges in very few iterations (sometimes 3-5 for small MDPs).
- Value iteration needs more outer iterations, but each is much cheaper (no inner loop). It can be viewed as doing "truncated policy evaluation" with just one sweep per iteration.
- For large state spaces, value iteration is often preferred because the inner loop of policy evaluation can be expensive.

---

### 2. What happens if the discount factor gamma is close to 0 or 1?

**gamma close to 0 (e.g., gamma = 0.01):**

The agent becomes extremely myopic -- it only cares about the immediate next reward and almost completely ignores all future consequences. Mathematically, the value function becomes:
```
V(s) ≈ E[r_0]    (since gamma^1 ≈ 0, gamma^2 ≈ 0, ...)
```

In the Cliff Walking environment, this has severe consequences:
- The goal is many steps away, so its reward is discounted to near zero.
- The agent has no incentive to navigate toward the goal because the discounted value of reaching it is negligible.
- The agent may wander randomly because all states look roughly equally valuable (all have immediate reward of -1).
- Convergence is very fast because distant states have no influence on current values -- the value function is essentially flat.

**gamma close to 1 (e.g., gamma = 0.99):**

The agent becomes far-sighted -- future rewards are weighted almost as heavily as immediate rewards. The value function integrates rewards over a long horizon:
```
V(s) = E[r_0 + 0.99*r_1 + 0.99^2*r_2 + ... ]
```

Effects:
- The agent plans optimal long-horizon paths. It properly accounts for the benefit of reaching the goal (avoiding further -1 penalties) and the catastrophic cost of falling into the cliff (-100).
- The value differences between states become very large in magnitude. For example, a state 10 steps from the goal has V ≈ -10, while a state 20 steps away has V ≈ -20. These differences propagate across the entire grid.
- Convergence becomes **slower** for two reasons:
  1. Value information must propagate across many states (the effective "planning horizon" is ~1/(1-gamma) steps, so gamma=0.99 means ~100 steps).
  2. Each iteration only propagates values one step, so you need many iterations for information to travel from the goal back to the start.
- In the undiscounted limit (gamma = 1), convergence is not guaranteed for general MDPs, though it works for episodic tasks like Cliff Walking where all policies eventually terminate.

**Practical insight:** gamma = 0.9 (used in the homework) gives a planning horizon of ~10 steps, which is sufficient for Cliff Walking (the grid is 4x12, so the optimal path is ~13 steps). gamma = 0.99 would also work but converge more slowly.

---

### 3. How does increasing the slip probability affect the optimal policy?

#### slip_chance = 0.0 (deterministic)

The environment is fully deterministic -- every action is executed exactly as intended. The optimal policy takes the **shortest possible path** to the goal:
- From the start (bottom-left), move right along row 2 (directly above the cliff)
- At column 11, move down to the goal

This path has length ~13 steps, giving a total return of ~-13. There is absolutely no risk because the agent has perfect control. Walking next to the cliff is perfectly safe.

#### slip_chance = 0.01 (low stochasticity)

With 1% slip probability, each intended action has a 1% chance of being replaced by a uniformly random action (one of the 4 directions).

Near the cliff (row 2), a slip could send the agent downward into the cliff, incurring -100. The expected penalty per step adjacent to the cliff is approximately:
```
0.01 * (1/4) * (-100) = -0.25 per step
```
(1% chance of slipping, 1/4 chance the random action is "down", -100 penalty)

Since the shortest path along the cliff is ~11 steps, the expected cliff penalty is ~11 * -0.25 = -2.75. The alternative safe path through the top row adds ~6 extra steps (-6 penalty). Since -2.75 < -6, the shortest path is still (barely) optimal. The policy is nearly identical to the deterministic case.

#### slip_chance = 0.2 (high stochasticity)

With 20% slip probability, the expected penalty per step near the cliff becomes:
```
0.2 * (1/4) * (-100) = -5.0 per step
```

Over ~11 cliff-adjacent steps, the expected penalty is ~-55. This massively exceeds the ~-6 penalty of taking the longer safe path. The optimal policy changes dramatically:
- From the start, the agent moves **up** (away from the cliff)
- In rows 0-1, the agent moves **right** along the top of the grid
- Near column 11, the agent moves **down** to the goal

The agent stays as far from the cliff as possible. Even in row 1, the agent prefers moving upward to row 0 before going right, because a slip in row 1 could push it to row 2 (adjacent to cliff), and another slip could push it into the cliff.

#### Why does the agent behave more conservatively as stochasticity increases?

The fundamental tradeoff is between **path length** and **cliff risk**:

```
Expected cost = (number of steps * -1) + (expected cliff penalties)
```

The path length cost scales linearly with extra steps (each adds -1). The cliff risk scales with:
```
risk ≈ (steps near cliff) * slip_chance * P(slip toward cliff) * (-100)
```

As slip_chance increases:
- The risk term grows proportionally
- At some threshold, the risk outweighs the benefit of the shorter path
- The optimal policy shifts from "shortest path" to "safest path"

This is a general principle in stochastic control: **under uncertainty, optimal policies become risk-averse**. The agent must account not just for what it intends to do, but for what might happen due to noise. Near dangerous states, the cost of rare bad outcomes (cliff falls) dominates the cost of slightly longer paths.

---

## Exercise 2: DQN

### 1. Why is experience replay important in DQN?

Experience replay addresses three critical problems with naive online Q-learning:

**Problem 1 -- Temporal correlation:** In online learning, consecutive transitions (s_t, a_t, r_t, s_{t+1}), (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}), ... are highly correlated. States are similar, actions follow the same policy, and rewards reflect the same part of the environment. Training a neural network on correlated data violates the i.i.d. assumption of SGD, causing the network to overfit to the most recent trajectory and develop biased gradients. For example, if the agent is stuck in one corner of the state space, it will only update Q-values for that corner, potentially "forgetting" what it learned about other regions.

**Problem 2 -- Sample efficiency:** Without replay, each transition is used exactly once for a gradient step and then discarded. With replay, transitions are stored and resampled many times, extracting more learning signal from each environment interaction. This is particularly important when environment interaction is expensive (e.g., real robots).

**Problem 3 -- Non-stationarity of the data distribution:** As the policy improves, the distribution of visited states changes. Without replay, the training data distribution shifts rapidly, which can destabilize learning. The replay buffer smooths this out by mixing old and new transitions, providing a more stable training distribution.

**How it works in practice:** The replay buffer stores the most recent N transitions (e.g., N=10,000). At each training step, a random mini-batch (e.g., 64 transitions) is sampled uniformly. Because the samples come from different episodes, different policies, and different parts of the state space, the mini-batch is approximately i.i.d., which is what SGD needs to converge.

---

### 2. What is the role of the target network in DQN? How does it improve stability?

**The problem without a target network:**

In Q-learning, the update target is:
```
y = r + gamma * max_a' Q(s', a'; theta)
```

Notice that the same parameters theta are used for both the prediction Q(s,a; theta) and the target max_a' Q(s', a'; theta). When we take a gradient step to move Q(s,a) toward y, we simultaneously change the target y itself (because theta changed). This creates a feedback loop:

1. Increase Q(s,a) toward the target
2. But the target also shifted (because theta changed)
3. So the new target might be even higher
4. Leading to more increases in Q(s,a)
5. Q-values can spiral upward or oscillate wildly

This is the "moving target" problem, and it makes training highly unstable -- especially with function approximation (neural networks), where a change to theta affects Q-values across all states simultaneously.

**How the target network fixes this:**

DQN maintains two networks:
- **Online network** Q(s,a; theta): Updated at every gradient step
- **Target network** Q(s,a; theta^-): A frozen copy, updated only every `target_update` steps

The TD target becomes:
```
y = r + gamma * max_a' Q(s', a'; theta^-)
```

Now the target is fixed for `target_update` steps, breaking the feedback loop. The online network can learn toward a stable objective. Periodically (e.g., every 50 steps), the target network is synchronized: theta^- <- theta.

**Effect on stability:** With the target network, the optimization landscape is locally stable for `target_update` steps. The network makes consistent progress toward a fixed target before that target shifts. This dramatically reduces oscillations and divergence.

**Trade-off:** If `target_update` is too large, the target becomes very stale, slowing learning. If too small (e.g., 1 = no target network), we get the instability described above. Typical values are 10-1000 depending on the problem.

---

### 3. What is Double DQN, and how does it reduce overestimation bias?

**The overestimation problem in standard DQN:**

Standard DQN computes the target as:
```
y = r + gamma * max_a' Q_target(s', a')
```

The `max` operator introduces a systematic **positive bias**. To understand why, consider that Q-values are estimates with noise (estimation error). For any state s', the estimated Q-values are:
```
Q(s', a_1) = Q*(s', a_1) + noise_1
Q(s', a_2) = Q*(s', a_2) + noise_2
...
```

Taking the max:
```
max_a Q(s', a) >= max_a Q*(s', a)
```

This is because the max operator preferentially selects actions where the noise happens to be positive. Even if all Q-values have zero-mean noise, the maximum of noisy estimates is biased upward. The more actions there are, and the noisier the estimates, the worse the overestimation.

This bias propagates through the Bellman backup: overestimated targets lead to overestimated Q-values, which lead to even more overestimated targets, potentially snowballing to very large Q-values.

**How Double DQN fixes this:**

Double DQN (van Hasselt et al., 2015) decouples action **selection** from action **evaluation** using the two networks that DQN already has:

```
a* = argmax_a' Q_online(s', a'; theta)       # select with online
y  = r + gamma * Q_target(s', a*; theta^-)    # evaluate with target
```

Compare with standard DQN:
```
y = r + gamma * max_a' Q_target(s', a'; theta^-)   # target selects AND evaluates
```

**Why this reduces overestimation:** The online network selects the action it thinks is best, but the target network independently evaluates that action. If the online network overestimates Q(s', a*) due to noise, the target network's estimate of the same action is unlikely to have the same noise (since they have different parameters). The target network provides a more neutral evaluation.

Mathematically, the overestimation requires that the selection and evaluation errors be correlated. By using different networks, we decorrelate them, and the bias shrinks toward zero.

**In practice:** Double DQN is a one-line code change from standard DQN (just change how `a*` is computed) but consistently produces better results, especially in environments where standard DQN's Q-values diverge.

---

## Exercise 3: PPO

### 1. Why does PPO clip the probability ratio instead of directly constraining the KL divergence like TRPO? What goes wrong if you remove clipping entirely?

**Why TRPO's approach is problematic:**

TRPO (Trust Region Policy Optimization) solves a constrained optimization problem at each step:
```
maximize E[ r(theta) * A ]
subject to KL(pi_old || pi_new) <= delta
```

This requires:
1. Computing the KL divergence constraint (requires the full distribution)
2. Computing the natural gradient using the Fisher information matrix (second-order derivatives)
3. Conjugate gradient solver to approximate the natural gradient
4. Line search to satisfy the constraint

This is complex to implement, computationally expensive (especially the second-order computation), and hard to extend to architectures with shared parameters between actor and critic.

**Why PPO clips instead:**

PPO replaces the hard KL constraint with a simple modification to the objective:
```
L_CLIP = E[ min( r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A ) ]
```

This achieves a similar effect using only first-order gradients (Adam), is trivial to implement, and naturally handles shared network architectures. The clipping creates a "soft trust region" -- the gradient vanishes when the ratio moves outside [1-eps, 1+eps], effectively stopping the policy from changing too much.

**Detailed mechanics of the clip:**
- When A > 0 (good action we want to reinforce): r can increase up to 1+eps, then the gradient is clipped to zero. This prevents over-committing to a single good action.
- When A < 0 (bad action we want to suppress): r can decrease down to 1-eps, then the gradient clips. This prevents over-suppressing.
- The `min` ensures we always take the more conservative (pessimistic) estimate.

**What goes wrong without clipping:**

Without clipping, the objective is just the standard importance-sampled policy gradient:
```
L = E[ r(theta) * A ]
```

The problem is that r(theta) = pi_new(a|s) / pi_old(a|s) can become very large. For example, if the new policy assigns 100x higher probability to some action than the old policy, r = 100. If this action happened to have a positive advantage, the gradient pushes to increase its probability even further.

This causes:
1. **Catastrophically large updates:** A single mini-batch with a large ratio can destroy the learned policy in one step.
2. **Policy collapse:** The policy can become nearly deterministic for one action, making recovery impossible.
3. **Reward hacking:** The importance weight amplifies rare, high-advantage actions disproportionately, even if they were flukes.

In our homework, the adaptive learning rate based on KL divergence provides an additional safety mechanism beyond clipping, but clipping is the primary stabilizer.

---

### 2. PPO throws away all collected data after each update. Why can't you simply reuse old rollouts for more gradient steps?

**The on-policy constraint:**

PPO is an on-policy algorithm: the data must come from the current policy for the gradient estimates to be valid. The policy gradient theorem says:
```
∇J(theta) = E_{tau ~ pi_theta}[ sum_t ∇log pi_theta(a_t|s_t) * A_t ]
```

The expectation is explicitly over trajectories from the current policy pi_theta. When we use importance sampling to reuse old data:
```
∇J ≈ E_{tau ~ pi_old}[ r(theta) * ∇log pi_theta(a_t|s_t) * A_t ]
```

This is only valid when pi_theta ≈ pi_old (so the importance weights r are close to 1).

**What happens with too many gradient steps:**

After each gradient step, theta changes, and pi_theta drifts further from pi_old. After enough steps:

1. **Importance weights become unreliable:** The ratios r(theta) = pi_new / pi_old can become very large or very small, making the variance of the gradient estimate explode. A few transitions with large weights dominate the entire gradient.

2. **Advantage estimates become stale:** The advantages A_t were computed using the old value function V^{pi_old}. After the policy changes, these advantages no longer accurately reflect which actions are better under the new policy.

3. **State distribution mismatch:** The old rollout visited states according to pi_old's state visitation distribution. The new policy pi_theta would visit a different distribution of states. The gradient update is biased toward states that pi_old visited but pi_theta might rarely encounter.

4. **Clipping stops helping:** The PPO clip prevents the ratio from moving beyond [1-eps, 1+eps], but after many gradient steps, most ratios hit the clip boundary, and the effective gradient becomes zero for most samples. The algorithm stops learning but hasn't converged.

**PPO's practical compromise:** Using n_epochs=10 with mini-batches is a careful balance. The clipping provides enough safety for ~10 passes over the data, but beyond that, the data is too stale. This is why n_epochs is typically 3-15, not 100.

**Contrast with SAC:** SAC can reuse old data because it uses the Q-function to evaluate actions, not importance-weighted policy gradients. The Q-function is trained to be correct for all (s,a) pairs, not just those visited by the current policy.

---

### 3. What does the GAE parameter lambda control? What happens at the extremes lambda = 0 and lambda = 1?

**Background -- the bias-variance tradeoff:**

When estimating advantages for policy gradients, we face a fundamental tradeoff:
- Using actual rewards (Monte Carlo) gives **unbiased** estimates but **high variance** (each trajectory is noisy)
- Using the learned value function V (bootstrapping) gives **lower variance** but introduces **bias** (V is imperfect)

GAE provides a smooth, tunable interpolation between these extremes.

**The GAE formula:**

Given TD residuals delta_t = r_t + gamma * V(s_{t+1}) - V(s_t), the GAE advantage is:
```
A_t^GAE = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
```

This is an exponentially-weighted sum of TD residuals, with decay rate (gamma * lambda).

**lambda = 0: Pure bootstrapping (1-step TD)**

```
A_t = delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

The advantage looks only one step ahead. It uses the immediate reward r_t and then bootstraps entirely from V(s_{t+1}).

- **Low variance:** Only one random reward r_t is involved. The estimate is very stable across different trajectories.
- **High bias:** If V is inaccurate (which it always is early in training), the advantage estimate is wrong. For example, if V(s_{t+1}) is overestimated, the advantage will be overestimated, potentially reinforcing bad actions.
- **Practical effect:** Learning is stable but can converge to suboptimal policies because the biased advantages point in slightly wrong directions.

**lambda = 1: Pure Monte Carlo (no bootstrapping)**

```
A_t = sum_{l=0}^{T-t} gamma^l * delta_{t+l}
     = (sum_{l=0}^{T-t} gamma^l * r_{t+l}) - V(s_t)
     = G_t - V(s_t)
```

This reduces to the full discounted return minus the baseline V(s_t). The advantage uses all actual future rewards with no bootstrapping from V (except as a baseline).

- **No bias:** The return G_t is the true discounted sum of rewards. Combined with any baseline, the policy gradient is unbiased.
- **High variance:** The return depends on all future rewards, each of which is random. In a 30-step episode, the variance comes from 30 random variables multiplied by 30 random actions. This can make the gradient signal very noisy.
- **Practical effect:** Learning is unbiased but very noisy. The agent needs many more samples to get a reliable gradient direction. Training curves are jagged.

**lambda = 0.95 (typical, used in the homework):**

The exponential weighting gives most weight to the first few TD residuals and exponentially less to distant ones:
- delta_t gets weight 1
- delta_{t+1} gets weight gamma * 0.95 ≈ 0.94
- delta_{t+5} gets weight (gamma * 0.95)^5 ≈ 0.73
- delta_{t+10} gets weight (gamma * 0.95)^10 ≈ 0.53
- delta_{t+20} gets weight (gamma * 0.95)^20 ≈ 0.28

So the estimate is dominated by the next ~10-20 steps, with diminishing contribution from further out. This gives:
- **Low bias:** Enough actual rewards are included to mostly overcome V's errors
- **Moderate variance:** Not as noisy as full Monte Carlo because distant (highly variable) rewards are down-weighted

This is why lambda = 0.95 is the default in most PPO implementations -- it works well across a wide range of tasks.

---

## Exercise 4: SAC

### 1. SAC adds an entropy bonus to the reward. What are the benefits of this?

SAC optimizes a maximum-entropy objective:
```
J(pi) = E[ sum_t gamma^t * (r_t + alpha * H(pi(·|s_t))) ]
```

where H(pi) = -E[log pi(a|s)] is the entropy of the policy. This has several concrete benefits:

**Benefit 1 -- Systematic exploration:**

Standard RL policies become deterministic as training progresses, which can trap the agent in local optima. With the entropy bonus, the policy is incentivized to maintain stochasticity. This means the agent continues to try diverse actions even in states where it has already found a good action.

For example, in the SO100 task, there might be multiple joint configurations that reach the target. Without entropy, the agent commits to the first one it finds. With entropy, it maintains probability on all good configurations, which is useful if the best one varies with the target position.

**Benefit 2 -- Robustness:**

A stochastic policy is inherently more robust to perturbations. If the environment dynamics change slightly (e.g., different friction, different target distribution at test time), a deterministic policy that was optimized for the exact training conditions may fail catastrophically. A stochastic policy that maintains coverage over multiple good actions is more likely to still perform well.

**Benefit 3 -- Better optimization landscape:**

The entropy bonus smooths the optimization landscape. Without it, the policy gradient can have sharp discontinuities when the policy switches between actions. The entropy term acts as a regularizer that keeps gradients well-behaved.

**Benefit 4 -- Composability and transfer:**

Maximum-entropy policies capture all near-optimal behaviors rather than committing to one. This is useful for transfer learning: a policy trained with entropy can be quickly fine-tuned for related tasks because it already "knows about" alternative action sequences.

**Benefit 5 -- Avoiding premature convergence:**

In continuous action spaces, it's easy for the policy's std to shrink to near-zero early in training, collapsing to a deterministic policy before finding a good solution. The entropy bonus penalizes this collapse, giving the agent more time to explore before committing.

---

### 2. SAC squashes actions through tanh. Why does this require a log-probability correction?

**Why tanh is used:**

The actor outputs a Gaussian distribution in unbounded space: u ~ N(mu, sigma^2), where u can be any real number. But the environment requires actions in [-1, 1]. The tanh function maps R -> (-1, 1), so a = tanh(u) is always valid.

**The change-of-variables problem:**

When we apply a nonlinear transformation a = tanh(u) to a random variable u, the probability density changes. Intuitively, tanh "compresses" large values of u (near +/- infinity) into a narrow range near +/- 1. This compression means the density of a near the boundaries is higher than you'd expect from the density of u alone.

Formally, the change-of-variables formula gives:
```
p(a) = p(u) * |du/da|
     = p(u) / |da/du|
     = p(u) / (1 - tanh(u)^2)
     = p(u) / (1 - a^2)
```

Taking logs:
```
log p(a) = log p(u) - log(1 - a^2)
```

Summing over action dimensions (since they're independent):
```
log pi(a|s) = sum_i [ log N(u_i; mu_i, sigma_i) - log(1 - a_i^2) ]
```

**Why this matters for SAC:**

SAC uses log pi(a|s) in three places:
1. **Critic target:** y = r + gamma * (1-d) * [min(Q1, Q2) - alpha * log pi(a'|s')]
2. **Actor loss:** L = alpha * log pi(a|s) - min(Q1, Q2)
3. **Alpha loss:** L = -log_alpha * (log pi(a|s) + H_target)

If we used the uncorrected log probability log p(u) instead of the corrected log p(a), all three would be wrong:
- The entropy estimate would be too high (because we'd be ignoring the density compression at the boundaries)
- Alpha would be tuned to the wrong level
- The actor would receive incorrect gradient signals

**Numerical detail:** The `1e-6` term in `log(1 - a^2 + 1e-6)` prevents log(0) when a is exactly +/- 1, which happens when u is very large.

---

### 3. The temperature alpha is tuned automatically. What happens when the policy's entropy is above vs. below the target?

**The alpha loss:**
```
L_alpha = E[ -log_alpha * (log pi(a|s) + H_target) ]
```

where H_target = -act_dim (in our case, -6). Note that entropy H = -E[log pi], so higher log pi means lower entropy.

**Case 1 -- Entropy is above target (policy is "too random"):**

If the policy's entropy is higher than |H_target|, then on average:
```
log pi(a|s) < -H_target    (log probs are very negative)
log pi(a|s) + H_target < 0
```

The gradient of L_alpha with respect to log_alpha is:
```
dL/d(log_alpha) = -(log pi + H_target)     [averaged over batch]
```

Since (log pi + H_target) < 0, the gradient is positive, so log_alpha **decreases** during optimization. This means alpha = exp(log_alpha) **decreases**.

**Effect:** Lower alpha means less entropy regularization. The policy is allowed to become more deterministic since it's already "random enough." The reward signal becomes more dominant in the actor loss.

**Case 2 -- Entropy is below target (policy is "too deterministic"):**

If entropy is below |H_target|:
```
log pi(a|s) > -H_target    (log probs are close to zero)
log pi(a|s) + H_target > 0
```

The gradient is negative, so log_alpha **increases**, and alpha **increases**.

**Effect:** Higher alpha means more entropy regularization. The entropy term alpha * log pi in the actor loss becomes larger, pushing the policy to increase its std and explore more broadly.

**The equilibrium:**

At convergence, the system reaches a balance where:
```
E[log pi(a|s)] ≈ -H_target = act_dim = 6
```

This means the policy maintains an entropy of approximately 6 nats (one nat per action dimension). The alpha oscillates around the value needed to maintain this entropy level. If the policy starts to collapse (e.g., because it found a good action sequence), alpha automatically increases to push it back toward the target entropy.

**Why use log_alpha instead of alpha directly:**

Optimizing log_alpha ensures that alpha = exp(log_alpha) is always positive, without needing an explicit constraint. It also provides more stable gradients because the learning rate acts in log-space, giving proportional updates (e.g., doubling alpha requires the same step size as halving it).

---

### 4. How does SAC compare with PPO in terms of update-to-data (UTD) ratio?

**Definition:** UTD = (number of gradient update steps) / (number of environment steps collected)

**PPO's UTD:**

PPO collects `n_steps = 2048` environment transitions, then performs `n_epochs = 10` passes over the data in mini-batches of `mini_batch_size = 1024`:
```
gradient_steps_per_iteration = n_epochs * (n_steps / mini_batch_size)
                              = 10 * (2048 / 1024)
                              = 20

UTD = 20 / 2048 ≈ 0.01
```

This means for every 100 environment steps, PPO performs approximately 1 gradient update. The UTD is very low because PPO is on-policy: it must discard data after the update and collect fresh transitions.

**SAC's UTD:**

SAC collects 1 environment transition, then performs 1 (or more) gradient updates:
```
UTD = gradient_steps / env_steps = 1/1 = 1
```

Some SAC variants use UTD > 1 (e.g., UTD = 20), performing multiple gradient steps per environment step. This is possible because SAC uses a replay buffer, so old data can be resampled.

**Comparison:**

| | PPO | SAC |
|---|---|---|
| UTD | ~0.01 | ~1 (or higher) |
| Env steps to solve | ~1M | ~100K |
| Gradient steps to solve | ~10K | ~100K |
| Compute per env step | Very low | Higher |

SAC is ~10x more sample-efficient (fewer env steps needed) but performs ~10x more gradient computations. In scenarios where environment interaction is expensive (real robots, expensive simulations), SAC's higher sample efficiency is a major advantage. When environment interaction is cheap (fast simulators), PPO's simplicity and parallelizability can be preferable.

---

### 5. Briefly discuss the advantages and disadvantages of on-policy vs. off-policy algorithms.

**On-policy algorithms (e.g., PPO, TRPO, A2C):**

*Advantages:*
- **Stability:** Data always matches the current policy, so gradient estimates are correct by construction. No approximation errors from importance sampling or off-policy correction. Training curves are typically smoother.
- **Simplicity:** No replay buffer, no target networks, no polyak averaging. Fewer hyperparameters to tune. Easier to debug.
- **Parallelizability:** Easy to scale by collecting data with multiple parallel workers. Each worker runs the same policy independently. This is how PPO achieves practical wall-clock speedups.
- **Reliable convergence:** On-policy methods have stronger theoretical convergence guarantees. The policy gradient theorem holds exactly for on-policy data.

*Disadvantages:*
- **Sample inefficiency:** Every transition is used only once (or for a few epochs). In our homework, PPO uses 1M environment steps. For real robots or expensive simulations, this is prohibitive.
- **Cannot leverage prior data:** Cannot use demonstrations, pre-collected datasets, or data from other policies. Each new training run starts from scratch.
- **Sensitive to step size:** Even with clipping, too-large updates can cause performance collapse from which on-policy methods cannot recover (the new bad policy generates bad data, leading to worse updates).

**Off-policy algorithms (e.g., SAC, TD3, DDPG):**

*Advantages:*
- **Sample efficiency:** The replay buffer allows each transition to be used for many gradient updates. SAC needs ~10x fewer environment steps than PPO for the same task.
- **Data reuse:** Can incorporate data from any source: old policies, demonstrations, other agents. This enables offline RL and learning from demonstrations.
- **Continuous learning:** Can learn without stopping to collect new data. The replay buffer provides a stable, diverse training distribution even as the policy changes.
- **Higher UTD:** Can perform many gradient steps per environment step, making better use of available compute.

*Disadvantages:*
- **Instability:** Off-policy data creates a distribution mismatch between the data and the current policy. This can cause the critic to extrapolate incorrectly in unvisited regions, leading to divergence. Mitigation requires target networks, double Q-learning, and careful hyperparameter tuning.
- **Complexity:** More moving parts: replay buffer, target networks (with polyak averaging), separate optimizers for actor/critic/alpha, entropy tuning. More hyperparameters (tau, buffer size, start_steps, alpha_lr, etc.). Harder to debug.
- **Stale data:** Old transitions in the replay buffer may no longer be relevant. In non-stationary environments, this can hurt performance. Prioritized replay and buffer size tuning help but add complexity.
- **Harder to parallelize:** The replay buffer is a shared resource. Scaling to multiple workers requires careful synchronization. Less naturally parallel than on-policy methods.

**When to use which:**

- **On-policy (PPO):** When environment interaction is cheap (fast simulators), stability is paramount, or you want simple, reliable training. Preferred for large-scale distributed training (e.g., OpenAI Five, Dota 2).
- **Off-policy (SAC):** When sample efficiency matters (real robots, expensive simulations), you have access to prior data, or you need continuous online learning. Preferred for robotics and real-world applications.
