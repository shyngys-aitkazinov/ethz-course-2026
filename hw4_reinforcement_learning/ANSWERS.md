# Homework 4: Theoretical Answers

---

## Exercise 1: Dynamic Programming

### 1. Policy Iteration vs Value Iteration

- **Policy Iteration**: Two phases per iteration -- full policy evaluation (iterates until V^pi converges), then one greedy policy improvement. Fewer outer iterations, but each is expensive.
- **Value Iteration**: Single Bellman optimality backup per iteration: `V(s) = max_a Q(s,a)`. More iterations, but each is cheap. Policy extracted once at the end.

### 2. Discount factor gamma

- **gamma -> 0**: Agent is myopic, only cares about immediate reward. Won't plan long paths to the goal.
- **gamma -> 1**: Agent is far-sighted, plans optimally over long horizons. But convergence slows down since value differences between states shrink.

### 3. Effect of slip probability

- **slip=0.0**: Shortest path along the cliff edge. No risk, so minimize steps.
- **slip=0.01**: Nearly the same -- small risk doesn't outweigh shorter path.
- **slip=0.2**: Agent detours via the top row, staying far from the cliff.

**Why more conservative?** Each step near the cliff risks a -100 penalty with probability proportional to slip_chance. At slip=0.2, the expected cost of walking near the cliff (~-20/step) far exceeds the extra -1/step penalties of the longer safe path.

---

## Exercise 2: DQN

### 1. Why is experience replay important in DQN?

Consecutive transitions are highly correlated (similar states, same policy). Training on correlated data causes the network to overfit to recent experience and forget earlier lessons. Experience replay breaks these correlations by sampling random mini-batches from a buffer of past transitions, making updates more stable and sample-efficient.

### 2. What is the role of the target network? How does it improve stability?

The target network provides a stable Q-value target for the Bellman update. Without it, both the prediction and the target shift simultaneously at every gradient step, creating a "moving target" problem that causes oscillations or divergence. By updating the target network only periodically (every `target_update` steps), the target stays fixed long enough for the online network to learn toward it.

### 3. What is Double DQN, and how does it reduce overestimation bias?

Standard DQN uses `max_a Q_target(s', a)` for the target, which overestimates Q-values because the same network both selects and evaluates the best action -- noise in Q-estimates biases the max upward.

Double DQN decouples selection and evaluation:
- **Select** the best action using the online network: `a* = argmax_a Q_online(s', a)`
- **Evaluate** it using the target network: `Q_target(s', a*)`

This reduces overestimation because the online network's noise in selecting `a*` is independent of the target network's noise in evaluating it.

---

## Exercise 3: PPO

### 1. Why does PPO clip instead of using a KL constraint like TRPO?

TRPO's hard KL constraint requires second-order optimization (conjugate gradient + line search), which is complex and expensive. PPO's clipping achieves a similar effect -- preventing the policy ratio from deviating too far from 1 -- using only first-order gradients (Adam). Without clipping, large ratios can cause catastrophically large policy updates that destroy the learned policy.

### 2. Why can't you reuse old rollouts for more gradient steps?

PPO is on-policy: the advantage estimates and log-probabilities are only valid under the policy that collected the data. After several gradient steps, the policy has changed, making the old data stale. The clipping helps for a few epochs, but eventually the importance-sampling ratio becomes too unreliable, leading to incorrect gradient estimates and unstable training.

### 3. What does GAE lambda control?

Lambda controls the bias-variance tradeoff in advantage estimation:
- **lambda = 0**: Pure 1-step TD advantage `A = r + gamma*V(s') - V(s)`. Low variance but high bias (depends on V accuracy).
- **lambda = 1**: Monte Carlo advantage (full return minus baseline). Unbiased but high variance.
- **lambda = 0.95** (typical): Mostly relies on nearby rewards with some longer-horizon signal. Good practical balance.

---

## Exercise 4: SAC

### 1. What are the benefits of the entropy bonus?

- **Better exploration**: Encourages the policy to remain stochastic, preventing premature convergence to a suboptimal deterministic policy.
- **Robustness**: Stochastic policies are more robust to perturbations and model errors.
- **Multi-modality**: The policy can maintain probability on multiple good actions rather than committing to one.

### 2. Why does tanh squashing require a log-probability correction?

Tanh is a nonlinear transformation applied to the sampled Gaussian action: `a = tanh(u)`. The change-of-variables formula requires a Jacobian correction:
```
log pi(a) = log pi(u) - sum log(1 - tanh(u)^2)
```
Without this correction, the log-probabilities would be wrong, leading to incorrect entropy estimates and broken actor/alpha updates.

### 3. What happens when entropy is above vs. below the target?

- **Entropy above target** (policy is "too random"): `logp + target_entropy < 0`, so alpha decreases. Less entropy regularization, allowing the policy to become more deterministic.
- **Entropy below target** (policy is "too certain"): `logp + target_entropy > 0`, so alpha increases. More entropy regularization pushes the policy to explore more.

This creates a self-regulating loop that maintains a desired level of stochasticity.

### 4. How does SAC compare with PPO in terms of UTD ratio?

- **PPO**: UTD ~ n_epochs * (n_steps / mini_batch_size) / n_steps. With typical settings (10 epochs, 2048 steps, 1024 batch), UTD ~ 20/2048 ~ 0.01. Very low -- each environment step is used for only a few gradient updates.
- **SAC**: UTD ~ 1 (or higher). Each new transition triggers one (or more) gradient updates on a mini-batch from the replay buffer. Much more gradient computation per environment step.

SAC is far more sample-efficient but requires more compute per step.

### 5. On-policy vs. off-policy: advantages and disadvantages

| | On-policy (PPO) | Off-policy (SAC) |
|---|---|---|
| **Sample efficiency** | Low -- data discarded after each update | High -- replay buffer reuses old data |
| **Stability** | More stable -- data always matches current policy | Can be unstable -- off-policy data may be stale |
| **Exploration** | Relies on policy stochasticity | Entropy bonus + replay diversity |
| **Hyperparameter sensitivity** | Generally more forgiving | More sensitive (critic LR, tau, alpha) |
| **Wall-clock time** | Fast per update, slow to converge | Slower per update, fewer env steps needed |
| **Implementation complexity** | Simpler (no replay buffer, no target networks) | More complex (replay buffer, target networks, 3 optimizers) |
