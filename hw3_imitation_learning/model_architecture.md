# ObstaclePolicy Architecture

## Overview

The `ObstaclePolicy` is an action-chunking MLP that maps a state vector `(B, state_dim)` to a chunk of future actions `(B, chunk_size, action_dim)`. It uses MSE loss for supervised regression against expert demonstrations.

```
state (B, 10)
    |
    v
[Linear(10 -> d_model)]
[LayerNorm(d_model)]        <- optional
[GELU]
    |
    v
[Linear(d_model -> d_model)]
[LayerNorm(d_model)]        <- optional
[GELU]
[Dropout(p)]                <- optional
    |
    v                        (repeat depth-1 times)
[Linear(d_model -> d_model)]
[LayerNorm(d_model)]        <- optional
[GELU]
[Dropout(p)]                <- optional
    |
    v
[Linear(d_model -> chunk_size * action_dim)]
    |
    v
reshape -> (B, chunk_size, action_dim)
```

## Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Hidden layer width |
| `depth` | 3 | Number of hidden layers (first layer has no dropout) |
| `dropout` | 0.1 | Dropout probability (0 to disable) |
| `activation` | `"gelu"` | Activation function: `relu`, `gelu`, `tanh`, `silu` |
| `use_layer_norm` | `True` | Whether to apply LayerNorm after each linear layer |
| `chunk_size` | 16 | Number of future actions predicted per forward pass |

## Parameter Count Examples

With `state_dim=10`, `action_dim=4`, `chunk_size=16` (output = 64):

| d_model | depth | Approx. params | Notes |
|---------|-------|----------------|-------|
| 128 | 2 | ~25K | Minimal, fast to train |
| 256 | 3 | ~200K | Recommended starting point |
| 256 | 4 | ~270K | Slightly more capacity |
| 512 | 3 | ~790K | Near the 1M limit |
| 512 | 4 | ~1.05M | Exceeds limit |

The constraint from the assignment is **< 1M parameters** for 100% SR on ex1/ex2.

## Why LayerNorm (and not BatchNorm)

### The core problem

The inputs to this MLP are z-normalized states (mean=0, std=1 at dataset level), but the **internal activations** at each hidden layer can drift to arbitrary scales during training. This is the "internal covariate shift" problem — each layer sees a shifting input distribution as earlier layers update their weights.

### Why LayerNorm helps here

1. **Consistent behavior at train and eval time.** LayerNorm normalizes across features *within each sample*. It computes mean and variance over the `d_model` dimension for each individual input independently. This means it behaves identically whether `batch_size=64` (training) or `batch_size=1` (evaluation/inference). This is critical because during policy evaluation, the model receives one state at a time.

2. **Stabilizes gradient flow.** Without normalization, deeper MLPs (depth >= 3) can suffer from vanishing or exploding gradients during training. LayerNorm keeps activations in a well-conditioned range, allowing the use of higher learning rates and faster convergence.

3. **Reduces sensitivity to initialization.** With LayerNorm, the network is more robust to the random weight initialization, leading to more reproducible training runs.

### Why NOT BatchNorm

BatchNorm normalizes across the *batch dimension* — it computes mean and variance over all samples in a mini-batch for each feature. This creates two problems for this task:

1. **Train/eval mismatch.** During training, BatchNorm uses mini-batch statistics. During eval, it uses running averages accumulated during training. With small datasets (30 episodes, ~28K samples), these running averages may not be representative, causing the model to behave differently at eval time. This is the most common source of "works during training, fails during eval" bugs.

2. **Batch size dependency.** At eval time, the policy processes one state at a time (`batch_size=1`). BatchNorm with a single sample is degenerate — the variance is zero, making the normalization undefined or numerically unstable. PyTorch handles this by using running stats in eval mode, but the fundamental issue of train/eval distribution mismatch remains.

3. **Stochastic behavior.** BatchNorm introduces noise through mini-batch statistics, which acts as a regularizer during training. This can be helpful for classification but is generally undesirable for regression tasks where we want deterministic, smooth predictions. Policy outputs need to be consistent — small state changes should produce small action changes.

### When to disable LayerNorm

- If your network is very shallow (depth=1-2) and d_model is small, LayerNorm adds overhead with minimal benefit
- If you're debugging and want to rule out normalization as a source of issues
- If you're using a very high dropout rate that already provides sufficient regularization

Set `use_layer_norm=False` to disable it and compare.

## Activation Function Choices

| Activation | Formula | Characteristics |
|------------|---------|-----------------|
| `relu` | `max(0, x)` | Simple, fast. Can cause "dead neurons" where gradients are permanently zero for negative inputs. |
| `gelu` | `x * Phi(x)` | Smooth approximation of ReLU. Used in transformers. Better gradient flow for regression since it doesn't hard-clip at zero. **Recommended.** |
| `silu` | `x * sigmoid(x)` | Also called Swish. Similar properties to GELU. Good alternative. |
| `tanh` | `tanh(x)` | Bounded output [-1, 1]. Can cause vanishing gradients in deep networks. Rarely the best choice. |

**Why GELU over ReLU for this task:** The policy predicts continuous action deltas. GELU's smooth, non-zero gradient everywhere means the model can make fine-grained adjustments during training. ReLU's hard zero below 0 can create flat loss landscapes where gradient-based optimization stalls.

## Dropout Considerations

- **0.0** — No regularization. Try this if your dataset is large or model is small.
- **0.1** — Light regularization. Good default for ~28K samples with ~200K params.
- **0.2-0.3** — Moderate. Use if you see large train/val loss gaps (overfitting).
- **> 0.3** — Aggressive. Likely hurts regression accuracy. Not recommended.

Note: Dropout is only applied after hidden layers (not the first or last layer). During eval, dropout is automatically disabled by `model.eval()`.

## Recommended Configurations

### Ex1 (single cube + obstacle, starting point)
```python
ObstaclePolicy(state_dim=10, action_dim=4, chunk_size=16,
               d_model=256, depth=3, dropout=0.1,
               activation="gelu", use_layer_norm=True)
# ~200K params
```

### Ex1 (if underfitting — need more capacity)
```python
ObstaclePolicy(state_dim=10, action_dim=4, chunk_size=16,
               d_model=512, depth=3, dropout=0.1,
               activation="gelu", use_layer_norm=True)
# ~790K params
```

### Ex1 (if overfitting — more regularization)
```python
ObstaclePolicy(state_dim=10, action_dim=4, chunk_size=16,
               d_model=256, depth=3, dropout=0.2,
               activation="gelu", use_layer_norm=True)
```

### Minimal (fast iteration)
```python
ObstaclePolicy(state_dim=10, action_dim=4, chunk_size=16,
               d_model=128, depth=2, dropout=0.0,
               activation="relu", use_layer_norm=False)
# ~25K params
```
