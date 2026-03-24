# Experiment Log — Exercise 1 (Single Cube with Obstacle)

## Dataset
- 4 recording sessions, 62 teleop episodes, ~15,002 timesteps
  - Session 1 (2026-03-23_23-10-22): 30 episodes, 9,066 timesteps
  - Session 2 (2026-03-23_23-51-41): 16 episodes, 3,499 timesteps
  - Session 3 (2026-03-24_02-20-16): 6 episodes, 1,236 timesteps
  - Session 4 (2026-03-24_16-19-15): 10 episodes, 1,263 timesteps — added to improve release behavior
- Action space: `ee` (3D xyz deltas) + absolute gripper
- State: `state_ee_xyz + state_gripper + state_cube + state_obstacle` (14D)
- Action: `action_ee_xyz + action_gripper` (4D)

### Data Quality Issues Found
- Many early episodes have NO gripper release — gripper only closes, never opens. Cube ends up in bin via gravity/positioning, not explicit release.
- Episodes that DO release have 18-98 "waste" steps of re-closing after release. Release never occurs in the last 16 timesteps.
- **Implication:** The model learns "keep closing when near bin" because both no-release episodes and post-release waste reinforce this.
- **Key insight:** Trimming the dataset made things WORSE (4-24% SR vs 86%). The "waste" steps actually help the model learn stable behavior near the bin. The fix is to record more episodes with clean release (Session 4), not trim existing data.

---

## Key Findings

### Action Space
- `ee` (3D xyz) is far superior to `ee_full` (6D pos+euler). Rotations dominate MSE loss and the model fails to learn position control.

### Training Dynamics
- **Val loss does NOT correlate with eval SR** — lower val_loss can mean overfitting and worse SR.
- **Sweet spot ~200 epochs** with CosineAnnealingLR. Longer training (300-500 epochs) overfits.
- CosineAnnealingWarmRestarts causes LR spikes at restart boundaries that hurt eval performance.
- `chunk_size=16` is critical — `chunk_size=8` gives 0% SR because individual deltas are too small.

---

## Experiment Results

| Run | Config Changes from Best | Epochs | Val Loss | Eval SR | Notes |
|-----|-------------------------|--------|----------|---------|-------|
| 1 | ee_full, lr=1e-3, cosine, batch=64 | 200 | - | **7%** | Rotation dominates MSE |
| 2 | ee_xyz, lr=1e-3, cosine, batch=64 | 500 | 0.54 | **0%** | Wrong LR, too many epochs |
| 3 | ee_xyz, cosine_restarts T0=100 | 300 | 0.542 | **80%** | First good result |
| 4 | + gripper_weight=2.0, dropout=0.2 | 300 | 0.567 | **78%** | Gripper weight didn't help |
| 5 | + clip_actions=5σ, gripper_wt=2.0 | 200 | 0.467 | **69%** | Clipping hurt — changed normalization |
| 6 | Run 3 config, 500 epochs | 500 | 0.530 | **59%** | Overfitting despite val_loss dropping |
| 7 | chunk_size=8 | 300 | 0.509 | **0-2%** | Deltas too small for 8-step chunks |
| 8 | no-LayerNorm + ReLU + dropout=0 | 300 | 0.580 | **0-2%** | Architecture matters |
| 9 | **CosineAnnealingLR (no restarts)** | 200 | 0.556 | **86%** | Best result — no LR spike at epoch 100 |
| 10 | dropout=0.05 | 200 | 0.552 | **80%** | Less regularization didn't help |
| 11 | seed=7 | 200 | 0.472 | **78%** | Seed variance |
| 12 | Trimmed dataset (release only) | 200 | 0.544 | **24%** | Removing post-release steps hurt |
| 13 | Trimmed dataset (aggressive) | 200 | 0.419 | **4%** | Removing too much data catastrophic |

---

## Best Config (86% SR)

```bash
python scripts/train.py \
    --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
    --state-keys state_ee_xyz state_gripper state_cube state_obstacle \
    --action-keys action_ee_xyz action_gripper \
    --policy obstacle \
    --chunk-size 16 \
    --optimizer adamw \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --scheduler cosine \
    --epochs 200 \
    --batch-size 128 \
    --d-model 256 \
    --depth 3 \
    --dropout 0.1 \
    --activation gelu \
    --seed 42
```

### Eval-during-training (epoch 50/100/150/200/250/300):
72% → 62% → 68% → **88%** → 72% → 80%

Peak at epoch 200 despite val_loss still decreasing — confirms overfitting after epoch 200.

---

## Next Steps
- Record 10-15 new episodes with explicit gripper release above the bin
- Train on combined original + new data
- Target: 90%+ SR
