#!/bin/bash
set -e

# Step 1: Compute actions (ee = 3D xyz)
python scripts/compute_actions.py --action-space ee

# Step 2: Train policy
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

# Step 3: Evaluate
# python scripts/eval.py --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt --headless --num-episodes 100

# Step 4: Generate submission
# python student_eval/run_eval.py --exercise 1 --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt
