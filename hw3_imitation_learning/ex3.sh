#!/bin/bash
# Exercise 3: Multicube Goal-Conditioned Policy

# Step 1: Compute actions from multicube teleop data
python scripts/compute_actions.py --action-space ee --datasets-dir ./datasets/raw/multi_cube

# Step 2: Train multicube policy
python scripts/train_multicube.py \
    --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
    --state-keys \
        state_ee_xyz \
        state_gripper \
        "original_pos_cube_red[:3]" \
        "original_pos_cube_green[:3]" \
        "original_pos_cube_blue[:3]" \
        state_goal \
        goal_pos \
    --action-keys action_ee_xyz action_gripper \
    --policy multitask --chunk-size 16 --optimizer adamw \
    --lr 3e-4 --weight-decay 1e-4 --scheduler cosine \
    --epochs 80 --batch-size 128 --d-model 256 --depth 3 \
    --dropout 0.1 --activation gelu --gripper-weight 1.0 \
    --dagger-weight 1.0 --seed 42 --eval-goal all --eval-every 10 --eval-episodes 100 --film

# Step 3: Evaluate best sim checkpoint
# python scripts/eval.py --checkpoint checkpoints/multi_cube/best_sim_model_ee_xyz_multitask_dagger25ep.pt --multicube --headless --num-episodes 100

# Step 4: Generate submission
# python student_eval/run_eval --exercise 3 --checkpoint checkpoints/multi_cube/best_sim_model_ee_xyz_multitask_dagger25ep.pt
