# Exercise 3: PPO (Proximal Policy Optimization)

## Algorithm Overview

PPO is an **on-policy** actor-critic algorithm for continuous control. It collects a batch of experience using the current policy, updates the policy and value function, then **discards all data** and collects fresh experience.

The key idea: constrain policy updates so the new policy doesn't deviate too far from the old one, measured by KL divergence. PPO achieves this via a **clipped surrogate objective** rather than a hard constraint (like TRPO).

### Training Loop (high level)
```
for each iteration:
    1. Collect n_steps transitions using current policy
    2. Compute advantages (GAE) and returns
    3. For n_epochs, sample mini-batches and update actor + critic
```

---

## Code Walkthrough: `ex3_ppo.py`

### `__init__`

- Creates a `GaussianActor` (outputs mean, learned fixed std) and a `ValueNet` (critic).
- Uses a **single combined Adam optimizer** for both actor and critic.
- Key hyperparams: `clip_ratio`, `target_kl`, `n_epochs`, `gae_lambda`.

### `select_action(obs)` — TODO

Called during rollout collection. Must:
1. `self.actor.act(obs)` — samples action from Gaussian, also sets up the internal distribution
2. `torch.clamp(action, -1, 1)` — clip to valid range
3. `self.actor.get_actions_log_prob(action)` — log pi(a|s), sum over action dims
4. `self.actor.action_mean` / `self.actor.action_std` — read from the distribution
5. `self.critic(obs)` — state value V(s)

**Important**: `act()` only returns the sampled action (no log_prob). You must call `get_actions_log_prob()` separately. The properties `action_mean`, `action_std`, `entropy` are available after `act()` calls `update_distribution()` internally.

### `compute_kl_mean(old_mu, old_std, mu, std)` — TODO

KL divergence between two diagonal Gaussians, per dimension:
```
KL_d = log(std / old_std) + (old_std^2 + (old_mu - mu)^2) / (2 * std^2) - 0.5
```
Sum over action dimensions, then average over the batch.

Used to adaptively adjust the learning rate — if KL is too high, slow down; if too low, speed up.

### `compute_surrogate_loss(logp, old_logp, adv)` — TODO

The PPO clipped objective:
```
ratio = exp(logp - old_logp)
clipped_ratio = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
loss = -mean(min(ratio * adv, clipped_ratio * adv))
```
The clipping prevents the ratio from moving too far from 1, limiting the policy change.

Scaled by `surrogate_loss_coeff` (1.0 in config).

### `compute_value_loss(val, old_val, ret)` — TODO

Clipped value loss (analogous to surrogate clipping):
```
unclipped = (val - ret)^2
clipped_val = old_val + clamp(val - old_val, -clip_ratio, clip_ratio)
clipped = (clipped_val - ret)^2
loss = mean(max(unclipped, clipped))
```
Scaled by `value_loss_coeff` (0.01 in config — critic loss is downweighted).

### `compute_entropy_loss(entropy)` — TODO

Entropy bonus to encourage exploration:
```
loss = -entropy_coeff * mean(entropy)
```
Negative because we want to **maximize** entropy (minimize negative entropy).

### `update(rollout_batch)` — TODO

The core update loop. For each mini-batch:
1. Compute KL divergence, adjust learning rate
2. Compute surrogate loss + value loss + entropy loss
3. Sum into total loss
4. Zero grad, backward, clip gradients (`max_grad_norm`), optimizer step

---

## Config: `ex3_ppo_config.py`

| Parameter | Value | Role |
|-----------|-------|------|
| `hidden_sizes` | [256, 128, 128] | MLP architecture for actor and critic |
| `total_iterations` | 500 | Number of collect-update cycles |
| `n_steps` | 2048 | Environment steps per iteration |
| `mini_batch_size` | 1024 | Samples per gradient update |
| `n_epochs` | 10 | Passes over collected data per iteration |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE bias-variance tradeoff |
| `surrogate_loss_coeff` | 1.0 | Weight for policy loss |
| `value_loss_coeff` | 0.01 | Weight for critic loss (small) |
| `entropy_coeff` | 0.005 | Weight for entropy bonus |
| `clip_ratio` | 0.2 | PPO clipping parameter |
| `learning_rate` | 3e-4 | Initial Adam learning rate |
| `target_kl` | 0.01 | KL target for adaptive LR |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |

### Key design choices

- **`value_loss_coeff = 0.01`**: Critic loss is heavily downweighted relative to actor loss. This is because the actor and critic share an optimizer, and the critic loss magnitude tends to be much larger.
- **`entropy_coeff = 0.005`**: Small entropy bonus prevents premature convergence to a deterministic policy.
- **`target_kl = 0.01`**: Conservative KL target — the adaptive LR will slow down learning if the policy changes too fast.
- **`gae_lambda = 0.95`**: High lambda gives lower-bias advantage estimates at the cost of higher variance. lambda=0 would be pure 1-step TD (low variance, high bias), lambda=1 would be Monte Carlo (no bias, high variance).

---

## GaussianActor API (from `rl/networks.py`)

Understanding the actor API is critical for implementing the TODOs:

| Method/Property | Returns | When available |
|----------------|---------|----------------|
| `act(obs)` | sampled action tensor | Always (also calls `update_distribution`) |
| `act_inference(obs)` | mean action (deterministic) | Always |
| `update_distribution(obs)` | None (sets internal state) | Always |
| `get_actions_log_prob(action)` | log prob (summed over dims) | After `update_distribution` |
| `action_mean` | distribution mean | After `update_distribution` |
| `action_std` | distribution std | After `update_distribution` |
| `entropy` | entropy (summed over dims) | After `update_distribution` |

The std is **state-independent** — it's a learned parameter vector `log_std`, not a network output. This differs from SAC's `SquashedGaussianActor` which has state-dependent std.
