#!/bin/bash
set -e

# Step 1: Check ex1 policy on adversarial obstacles
# python scripts/eval.py --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt --adversarial --num-episodes 20 --speed 3

# Step 2: DAgger — take over when policy fails on adversarial obstacles
# python scripts/dagger_eval.py --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt --num-episodes 20

# Step 3: Re-compute actions (merges teleop + dagger data automatically)
python scripts/compute_actions.py --action-space ee

# Step 4: Retrain on combined data
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
    --epochs 100 \
    --batch-size 128 \
    --d-model 256 \
    --depth 3 \
    --dropout 0.1 \
    --activation gelu \
    --gripper-weight 2.0 \
    --seed 42

# Step 5: Evaluate on adversarial
# python scripts/eval.py --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger27ep.pt --adversarial --headless --num-episodes 100

# Step 6: Generate submission
# python student_eval/run_eval.py --exercise 2 --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger27ep.pt
