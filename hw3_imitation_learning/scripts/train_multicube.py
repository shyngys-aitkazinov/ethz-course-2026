"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import re
import subprocess
import sys

import numpy as np
import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
EPOCHS = 100
BATCH_SIZE = 128
LR = 3e-4
VAL_SPLIT = 0.1


def run_sim_eval(
    checkpoint_path: Path,
    num_episodes: int = 50,
    goal_cube: str = "all",
) -> float | None:
    """Run eval.py as subprocess, parse and return success rate."""
    cmd = [
        sys.executable,
        "scripts/eval.py",
        "--checkpoint",
        str(checkpoint_path),
        "--multicube",
        "--headless",
        "--num-episodes",
        str(num_episodes),
        "--goal-cube",
        goal_cube,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout + result.stderr
        # Parse final line: "Evaluation complete. Success rate: 32/50 (64%)"
        matches = re.findall(r"Success rate:\s*(\d+)/(\d+)\s*\((\d+)%\)", output)
        if matches:
            sr = int(matches[-1][2])
            return sr
    except subprocess.TimeoutExpired:
        print("  Eval timed out.")
    except Exception as e:
        print(f"  Eval error: {e}")
    return None


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="*",
        default=[],
        help="Additional .zarr stores to merge.",
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="multitask",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: multitask).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=[
            "state_ee_xyz",
            "state_gripper",
            "original_pos_cube_red[:3]",
            "original_pos_cube_green[:3]",
            "original_pos_cube_blue[:3]",
            "state_goal",
            "goal_pos",
        ],
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=["action_ee_xyz", "action_gripper"],
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate (default: {LR}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer: adam or adamw (default: adamw).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "cosine_restarts", "constant", "step", "plateau"],
        default="cosine",
        help="LR scheduler: cosine, cosine_restarts, constant, step, plateau (default: cosine).",
    )
    parser.add_argument(
        "--cosine-t0",
        type=int,
        default=10,
        help="Cycle length for cosine_restarts scheduler (default: 10).",
    )
    # ── model architecture args ──────────────────────────────────────
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Hidden layer width (default: 256).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Number of hidden layers (default: 3).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability (default: 0.1).",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu", "tanh", "silu"],
        default="gelu",
        help="Activation function (default: gelu).",
    )
    parser.add_argument(
        "--layer-norm",
        action="store_true",
        help="Enable LayerNorm in the MLP (disabled by default).",
    )
    parser.add_argument(
        "--gripper-weight",
        type=float,
        default=1.0,
        help="Loss weight for gripper action dimension (default: 1.0).",
    )
    parser.add_argument(
        "--film",
        action="store_true",
        default=True,
        help="Enable FiLM goal conditioning in MultiTaskPolicy (enabled by default).",
    )
    parser.add_argument(
        "--goal-dim",
        type=int,
        default=3,
        help="Dimension of the goal one-hot vector (last N dims of state). Only used with --film (default: 3).",
    )
    parser.add_argument(
        "--dagger-weight",
        type=float,
        default=1.0,
        help="Loss sampling weight for DAgger episodes relative to teleop (default: 1.0 = no upweighting).",
    )
    parser.add_argument(
        "--critical-weight",
        type=float,
        default=1.0,
        help="Duplication factor for critical phase samples (approach+release near bin). "
        "E.g. 3.0 = 3x copies of critical samples in train (default: 1.0 = no upweighting).",
    )
    parser.add_argument(
        "--critical-window",
        type=int,
        default=30,
        help="Number of steps before gripper release to include in critical window (default: 30).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Run sim eval every N epochs (0 = disabled). Runs eval.py as subprocess.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes for periodic sim eval (default: 100).",
    )
    parser.add_argument(
        "--eval-goal",
        type=str,
        default="all",
        choices=["red", "green", "blue", "all"],
        help="Goal color for sim eval (default: all).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # Snapshot original ep_ends before any augmentation (used by critical phase detection)
    orig_ep_ends = ep_ends.copy()

    # ── cube permutation augmentation (6x data) ─────────────────────
    # For multicube: permute cube position columns and goal one-hot to
    # create synthetic episodes for all 6 color permutations.
    # Requires state_keys to contain cube position + state_goal keys.
    if args.policy == "multitask" and args.state_keys:
        # Find column offsets for the 3 cube position arrays and goal one-hot
        from itertools import permutations

        _zroot = zarr_lib.open_group(str(args.zarr), mode="r")
        _zdata = _zroot["data"]
        from hw3.dataset import _parse_key_spec

        cube_key_names = ["original_pos_cube_red", "original_pos_cube_green", "original_pos_cube_blue"]
        cube_col_ranges: list[tuple[int, int]] = []  # (start, end) for each cube
        goal_col_range: tuple[int, int] | None = None
        col = 0
        for spec in args.state_keys:
            name, col_slice = _parse_key_spec(spec)
            arr_dim = np.asarray(_zdata[name][:1]).shape[1]
            # Apply slice to get actual width
            width = len(range(*col_slice.indices(arr_dim)))
            if name in cube_key_names:
                cube_col_ranges.append((col, col + width))
            elif name == "state_goal":
                goal_col_range = (col, col + width)
            col += width

        if len(cube_col_ranges) == 3 and goal_col_range is not None:
            # Generate all 6 permutations (including identity)
            perms = list(permutations(range(3)))
            orig_states = states.copy()
            orig_actions = actions.copy()

            all_states = [orig_states]  # identity perm already included
            all_actions = [orig_actions]
            all_ep_ends_list = [orig_ep_ends]

            for perm in perms:
                if perm == (0, 1, 2):
                    continue  # skip identity
                aug = orig_states.copy()
                # Permute cube columns: slot i gets data from slot perm[i]
                for dst_idx, src_idx in enumerate(perm):
                    dst_s, dst_e = cube_col_ranges[dst_idx]
                    src_s, src_e = cube_col_ranges[src_idx]
                    aug[:, dst_s:dst_e] = orig_states[:, src_s:src_e]
                # Permute goal one-hot
                gs, ge = goal_col_range
                for dst_idx, src_idx in enumerate(perm):
                    aug[:, gs + dst_idx] = orig_states[:, gs + src_idx]

                all_states.append(aug)
                all_actions.append(orig_actions.copy())
                # Shift episode ends
                offset = all_ep_ends_list[-1][-1]
                all_ep_ends_list.append(orig_ep_ends + offset)

            states = np.concatenate(all_states, axis=0)
            actions = np.concatenate(all_actions, axis=0)
            ep_ends = np.concatenate(all_ep_ends_list, axis=0)
            print(f"Cube permutation augmentation: 6 permutations, "
                  f"{orig_states.shape[0]} -> {states.shape[0]} steps, "
                  f"{len(orig_ep_ends)} -> {len(ep_ends)} episodes")
        else:
            print("Cube permutation augmentation: skipped (cube/goal keys not found)")

    normalizer = Normalizer.from_data(states, actions)

    # ── build dataset and split BEFORE any duplication ──────────────
    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")

    # ── train / val split (on original data) ─────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    # ── duplicate dagger samples in train only ───────────────────────
    dagger_repeat = int(args.dagger_weight)
    if n_dagger_eps > 0 and dagger_repeat > 1:
        n_total_eps = len(ep_ends)
        dagger_start_step = (
            int(ep_ends[n_total_eps - n_dagger_eps - 1])
            if n_dagger_eps < n_total_eps
            else 0
        )
        # Find which train indices point to dagger timesteps
        dagger_train_indices = [
            i for i in train_ds.indices if dataset.indices[i] >= dagger_start_step
        ]
        extra_copies = dagger_repeat - 1
        # Append extra copies of dagger indices to the train set
        from torch.utils.data import ConcatDataset, Subset

        dagger_subsets = [Subset(dataset, dagger_train_indices)] * extra_copies
        train_ds = ConcatDataset([train_ds] + dagger_subsets)
        print(
            f"DAgger upweighting: {len(dagger_train_indices)} dagger samples in train "
            f"duplicated {extra_copies}x (train size: {len(train_ds)})"
        )

    # ── duplicate critical-phase samples (approach + drop near bin) ─
    critical_repeat = int(args.critical_weight)
    if critical_repeat > 1:
        # Load cube positions and goal info from zarr for drop detection
        _zroot = zarr_lib.open_group(str(args.zarr), mode="r")
        _zdata = _zroot["data"]
        _goal_onehot = np.asarray(_zdata["state_goal"][:])
        _goal_pos = np.asarray(_zdata["goal_pos"][:])
        _ee_xyz = np.asarray(_zdata["state_ee_xyz"][:])
        _cube_arrays = {
            0: np.asarray(_zdata["original_pos_cube_red"][:, :3]),
            1: np.asarray(_zdata["original_pos_cube_green"][:, :3]),
            2: np.asarray(_zdata["original_pos_cube_blue"][:, :3]),
        }

        # Use original (pre-augmentation) ep_ends so indices match zarr row count
        _orig_ep_starts = np.concatenate([[0], orig_ep_ends[:-1]])
        critical_timesteps = set()
        BIN_THRESH = 0.06
        EE_CUBE_THRESH = 0.08
        n_detected = 0

        for s, e in zip(_orig_ep_starts, orig_ep_ends):
            s, e = int(s), int(e)
            color_idx = int(_goal_onehot[s].argmax())
            ep_cube = _cube_arrays[color_idx][s:e]
            ep_ee = _ee_xyz[s:e]
            ep_gp = _goal_pos[s]

            # Find when goal cube reaches the bin
            cube_to_bin = np.linalg.norm(ep_cube[:, :2] - ep_gp[:2], axis=1)
            in_bin = np.where(cube_to_bin < BIN_THRESH)[0]
            if len(in_bin) == 0:
                continue
            drop_step = int(in_bin[0])

            # Walk backwards: find last moment EE was near cube before drop
            if drop_step == 0:
                continue
            ee_to_cube = np.linalg.norm(
                ep_ee[:drop_step, :2] - ep_cube[:drop_step, :2], axis=1
            )
            near_cube = np.where(ee_to_cube < EE_CUBE_THRESH)[0]
            if len(near_cube) == 0:
                continue
            release_step = int(near_cube[-1])

            # Critical window: approach before release + drop
            win_start = max(0, release_step - args.critical_window)
            win_end = min(e - s, drop_step + 10)
            for t in range(win_start, win_end):
                critical_timesteps.add(s + t)
            n_detected += 1

        # Find which train indices fall in critical timesteps
        if hasattr(train_ds, "datasets"):
            base_train_indices = list(train_ds.datasets[0].indices)
        else:
            base_train_indices = list(train_ds.indices)

        critical_train_indices = [
            i for i in base_train_indices if dataset.indices[i] in critical_timesteps
        ]

        if critical_train_indices:
            extra = critical_repeat - 1
            from torch.utils.data import ConcatDataset, Subset
            critical_subsets = [Subset(dataset, critical_train_indices)] * extra
            if isinstance(train_ds, ConcatDataset):
                train_ds = ConcatDataset(list(train_ds.datasets) + critical_subsets)
            else:
                train_ds = ConcatDataset([train_ds] + critical_subsets)
            print(
                f"Critical-phase upweighting: {n_detected} eps detected, "
                f"{len(critical_train_indices)} samples duplicated {extra}x "
                f"(train size: {len(train_ds)})"
            )
        else:
            print("Critical-phase: no drop events detected, skipping.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
        activation=args.activation,
        use_layer_norm=args.layer_norm,
        gripper_weight=args.gripper_weight,
        use_film=args.film,
        goal_dim=args.goal_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── optimizer ──────────────────────────────────────────────────────
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # ── scheduler ─────────────────────────────────────────────────────
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_t0
        )
    elif args.scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.1
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    print(f"Optimizer: {args.optimizer} (lr={args.lr}, wd={args.weight_decay})")
    print(f"Scheduler: {args.scheduler}")

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_sr = -1
    sim_save_name = save_name.replace("best_model_", "best_sim_model_")
    sim_save_path = ckpt_dir / sim_save_name

    def _save_checkpoint(path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "normalizer": {
                    "state_mean": normalizer.state_mean,
                    "state_std": normalizer.state_std,
                    "action_mean": normalizer.action_mean,
                    "action_std": normalizer.action_std,
                },
                "chunk_size": args.chunk_size,
                "policy_type": args.policy,
                "state_keys": args.state_keys,
                "action_keys": args.action_keys,
                "state_dim": int(states.shape[1]),
                "action_dim": int(actions.shape[1]),
                "d_model": args.d_model,
                "depth": args.depth,
                "dropout": args.dropout,
                "activation": args.activation,
                "use_layer_norm": args.layer_norm,
                "use_film": args.film,
                "goal_dim": args.goal_dim,
                "val_loss": val_loss,
            },
            path,
        )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(save_path, epoch, val_loss)
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

        # Periodic sim eval
        if args.eval_every > 0 and epoch % args.eval_every == 0 and save_path.exists():
            print(
                f"  Running sim eval ({args.eval_episodes} eps, goal={args.eval_goal})..."
            )
            sr = run_sim_eval(save_path, args.eval_episodes, args.eval_goal)
            if sr is not None:
                if sr > best_sr:
                    best_sr = sr
                    _save_checkpoint(sim_save_path, epoch, val_loss)
                    print(f"  >>> Sim eval SR: {sr}% ✓ best sim checkpoint saved")
                else:
                    print(f"  >>> Sim eval SR: {sr}% (best: {best_sr}%)")

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Val checkpoint:     {save_path}")
    if sim_save_path.exists():
        print(f"Sim checkpoint:     {sim_save_path}  (best SR: {best_sr}%)")

    # Final eval
    if save_path.exists():
        print(f"\n{'=' * 50}")
        print(f"Final eval ({args.eval_episodes} eps, goal={args.eval_goal})...")
        sr = run_sim_eval(save_path, args.eval_episodes, args.eval_goal)
        if sr is not None:
            print(f"Final SR: {sr}%")


if __name__ == "__main__":
    main()
