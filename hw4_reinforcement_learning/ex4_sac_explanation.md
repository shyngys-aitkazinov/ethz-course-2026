# Exercise 4: SAC (Soft Actor-Critic)

## Algorithm Overview

SAC is an **off-policy** actor-critic algorithm that adds an **entropy bonus** to the reward. This encourages exploration and leads to more robust policies.

Key differences from PPO:
- **Off-policy**: stores transitions in a replay buffer, reuses old data
- **Entropy regularization**: maximizes `reward + alpha * entropy`
- **Double Q-learning**: two critics to reduce overestimation
- **Automatic temperature tuning**: alpha adapts to keep entropy near a target

---

## Code Walkthrough: `ex4_sac.py`

### Architecture

- **Actor**: `SquashedGaussianActor` — outputs state-dependent mean AND std (unlike PPO's fixed std). Actions are squashed through `tanh` to stay in [-1, 1]. `act()` returns `(action, log_prob)` with the tanh correction already applied.
- **Critic**: `DoubleQNet` — two independent Q-networks. `critic(obs, act)` returns `(q1, q2)`.
- **Target Critic**: Slowly-updated copy of the critic (Polyak averaging).
- **Three separate optimizers**: actor, critic, and alpha (temperature).

### `sample_action(obs)` — TODO

Simple: call `self.actor.act(obs)`, return only the action (discard log_prob).

### `compute_critic_loss(obs, act, rew, next_obs, done)` — TODO

Bellman target with entropy:
```
next_action, next_logp = self.actor.act(next_obs)
q1_next, q2_next = self.critic_target(next_obs, next_action)
q_next = min(q1_next, q2_next) - alpha * next_logp
target_q = rew + gamma * (1 - done) * q_next
```
Then MSE loss for both Q-networks against the target:
```
critic_loss = MSE(q1_pred, target_q) + MSE(q2_pred, target_q)
```

Key details:
- Use `self.critic_target` (not `self.critic`) for next-state Q-values
- `torch.min(q1_next, q2_next)` — pessimistic estimate reduces overestimation
- Subtract `alpha * next_logp` — entropy regularization in the target

### `compute_actor_loss(obs, act_new, logp_new)` — TODO

The actor wants to maximize Q-values while maintaining entropy:
```
q1_new, q2_new = self.critic(obs, act_new)
q_new = min(q1_new, q2_new)
actor_loss = mean(alpha * logp_new - q_new)
```
Minimizing `alpha * log_prob - Q` = maximizing `Q - alpha * log_prob` = maximizing reward + entropy.

### `compute_alpha_loss(logp_new)` — TODO

Automatic temperature tuning:
```
alpha_loss = mean(-log_alpha * (logp_new + target_entropy).detach())
```
- If entropy > target: `logp + target_entropy < 0`, so alpha decreases (less regularization needed)
- If entropy < target: `logp + target_entropy > 0`, so alpha increases (push for more exploration)
- `.detach()` is critical — alpha gradient should not flow through the actor

### `soft_update_targets()` — TODO

Polyak averaging (slow target update):
```
target_param = (1 - tau) * target_param + tau * online_param
```
Typically `tau = 0.005` — the target moves very slowly toward the online network.

### `update(batch)` — TODO

Sequential update order matters:
1. **Critic**: compute critic_loss, backprop, step
2. **Actor**: sample new actions from actor, compute actor_loss, backprop, step
3. **Alpha**: compute alpha_loss from the same logp_new, backprop, step
4. **Soft-update** target critics

---

## Key Differences: SAC vs PPO

| | PPO (Ex3) | SAC (Ex4) |
|---|---|---|
| On/off-policy | On-policy (discard data after update) | Off-policy (replay buffer) |
| Std | Fixed learned parameter | State-dependent network output |
| Action squashing | Clip to [-1,1] | tanh (with log-prob correction) |
| Exploration | Entropy bonus (small) | Entropy is core objective |
| Optimizers | Single combined | Three separate |
| Target network | None | Polyak-averaged critic target |
| Sample efficiency | Low (data used once) | High (data reused many times) |
