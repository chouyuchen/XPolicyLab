#!/usr/bin/env python3
"""Offline action-chunk analysis on the initial window of training episodes.

Loads a trained DP checkpoint, takes the first valid observation window
(n_obs_steps=2) from each specified training episode, runs policy inference
multiple times without environment stepping, and compares the executed
8-step action chunk to ground truth.

Purpose: does the policy correctly reproduce the initial approach action
on training episodes?

Does not modify training or deployment code.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset  # noqa: E402
from diffusion_policy.workspace.robotworkspace import RobotWorkspace  # noqa: E402

DEFAULT_CKPT = (
    SCRIPT_DIR
    / "checkpoints/cabinet_chain_nominal_dr-skill_close_loaded_drawer-aloha_agilex-joint-0/150.ckpt"
)
DEFAULT_ZARR = (
    SCRIPT_DIR / "data/cabinet_chain_nominal_dr-skill_close_loaded_drawer-aloha_agilex-joint.zarr"
)
# Same 5 demos used in training-init-replay (picker_seed=20260822).
DEFAULT_EPISODES = [27, 32, 135, 146, 196]

ACTION_LABELS = [
    "L_j0", "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_grip",
    "R_j0", "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_grip",
]


@dataclass
class EpisodeResult:
    episode_id: int
    window_idx: int
    mae_per_sample: list[float]
    mae_mean: float
    sampling_var_mean: float
    sampling_var_per_dim: list[float]


def resolve_ckpt(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(
            (int(p.stem), p) for p in path.glob("*.ckpt") if p.stem.isdigit()
        )
        if not candidates:
            raise FileNotFoundError(f"No numeric checkpoints under {path}")
        return candidates[-1][1]
    raise FileNotFoundError(path)


def load_policy(ckpt_path: Path, device: torch.device):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: RobotWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return policy, cfg


def build_dataset(cfg, zarr_path: Path) -> RobotImageDataset:
    return RobotImageDataset(
        zarr_path=str(zarr_path),
        horizon=int(cfg.horizon),
        pad_before=int(cfg.n_obs_steps) - 1,
        pad_after=int(cfg.n_action_steps) - 1,
        seed=int(cfg.task.dataset.seed),
        val_ratio=float(cfg.task.dataset.val_ratio),
        batch_size=1,
        max_train_episodes=cfg.task.dataset.max_train_episodes,
    )


def episode_bounds(dataset: RobotImageDataset) -> list[tuple[int, int]]:
    """Return (start, end) exclusive buffer indices for each episode."""
    ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
    bounds = []
    start = 0
    for end in ends:
        bounds.append((int(start), int(end)))
        start = int(end)
    return bounds


def first_window_index_for_episode(dataset: RobotImageDataset, episode_id: int) -> int:
    """Sampler index of the first valid window in a training episode.

    With pad_before = n_obs_steps - 1, the first window starts at the episode's
    first frame (obs padded for the missing history step).
    """
    bounds = episode_bounds(dataset)
    if episode_id < 0 or episode_id >= len(bounds):
        raise ValueError(
            f"episode_id={episode_id} out of range [0, {len(bounds) - 1}]"
        )
    if not bool(dataset.train_mask[episode_id]):
        raise ValueError(f"episode_id={episode_id} is in the validation split")

    ep_start, ep_end = bounds[episode_id]
    indices = dataset.sampler.indices  # (N, 4): buffer_start, buffer_end, sample_start, sample_end

    matches = np.where(
        (indices[:, 0] >= ep_start)
        & (indices[:, 1] <= ep_end)
        & (indices[:, 0] == ep_start)
    )[0]
    if len(matches) == 0:
        # Fallback: any window whose buffer start is inside the episode; take earliest.
        matches = np.where((indices[:, 0] >= ep_start) & (indices[:, 0] < ep_end))[0]
        if len(matches) == 0:
            raise RuntimeError(f"No sampler windows found for episode {episode_id}")
        matches = matches[np.argsort(indices[matches, 0])]

    return int(matches[0])


def sample_to_policy_obs(sample: dict, n_obs_steps: int, device: torch.device) -> dict[str, torch.Tensor]:
    def cam(key: str) -> torch.Tensor:
        # zarr stores cameras as T, 3, H, W (see process_data.py)
        arr = sample[key][:n_obs_steps].float() / 255.0
        return arr.unsqueeze(0).to(device)

    agent_pos = sample["state"][:n_obs_steps].float().unsqueeze(0).to(device)
    return {
        "head_cam": cam("head_camera"),
        "agent_pos": agent_pos,
    }


def ground_truth_chunk(sample: dict, n_obs_steps: int, n_action_steps: int) -> np.ndarray:
    start = n_obs_steps - 1
    end = start + n_action_steps
    return sample["action"][start:end].cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_executed_chunk(policy, obs_dict: dict[str, torch.Tensor], n_action_steps: int) -> np.ndarray:
    out = policy.predict_action(obs_dict)
    return out["action"][0, :n_action_steps].detach().cpu().numpy().astype(np.float32)


def parse_episodes(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("--episodes must list at least one episode id")
    return [int(p) for p in parts]


def plot_episode(
    out_path: Path,
    episode_id: int,
    window_idx: int,
    gt: np.ndarray,
    preds: np.ndarray,
    action_labels: list[str],
) -> None:
    n_steps, n_dim = gt.shape
    n_cols = 7
    n_rows = int(np.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)
    steps = np.arange(n_steps)

    for dim in range(n_dim):
        ax = axes[dim // n_cols, dim % n_cols]
        ax.plot(steps, gt[:, dim], "k-o", linewidth=2, markersize=4, label="GT")
        for s in range(preds.shape[0]):
            ax.plot(
                steps,
                preds[s, :, dim],
                alpha=0.55,
                linewidth=1.2,
                label=f"pred{s}" if dim == 0 else None,
            )
        ax.set_title(action_labels[dim], fontsize=9)
        ax.grid(True, alpha=0.25)
        if dim // n_cols == n_rows - 1:
            ax.set_xlabel("step")
    for dim in range(n_dim, n_rows * n_cols):
        axes[dim // n_cols, dim % n_cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=6, fontsize=8)
    fig.suptitle(
        f"episode {episode_id} initial window (sampler idx={window_idx}): "
        f"GT vs {preds.shape[0]} diffusion samples",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_summary(
    out_path: Path,
    mae_per_dim: np.ndarray,
    var_per_dim: np.ndarray,
    action_labels: list[str],
    n_episodes: int,
    n_samples: int,
) -> None:
    x = np.arange(len(action_labels))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, mae_per_dim, width=width, color="#4C72B0", label="MAE (pred mean vs GT)")
    ax2.bar(x + width / 2, var_per_dim, width=width, color="#DD8452", alpha=0.85, label="Sampling variance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(action_labels, rotation=45, ha="right")
    ax1.set_ylabel("MAE")
    ax2.set_ylabel("Variance across samples")
    ax1.set_title(
        f"Initial-window MAE / sampling variance "
        f"({n_episodes} episodes, {n_samples} samples each)"
    )
    ax1.grid(True, axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze DP initial-window action chunks on specified training episodes."
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    parser.add_argument(
        "--episodes",
        type=str,
        default=",".join(str(e) for e in DEFAULT_EPISODES),
        help="Comma-separated training episode ids (default: 27,32,135,146,196)",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    episode_ids = parse_episodes(args.episodes)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_ckpt(args.ckpt)
    zarr_path = args.zarr.expanduser().resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else SCRIPT_DIR / "data/outputs/action_chunk_analysis_initial" / timestamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, cfg = load_policy(ckpt_path, device)
    dataset = build_dataset(cfg, zarr_path)
    n_obs = int(cfg.n_obs_steps)
    n_action = int(cfg.n_action_steps)

    episode_to_window = {
        ep: first_window_index_for_episode(dataset, ep) for ep in episode_ids
    }

    print(f"[analyze] ckpt={ckpt_path}")
    print(f"[analyze] zarr={zarr_path}")
    print(f"[analyze] episodes={episode_ids}")
    print(f"[analyze] initial windows={episode_to_window}")
    print(f"[analyze] n_obs_steps={n_obs}, executed_chunk={n_action}, samples/episode={args.num_samples}")

    all_gt = []
    all_pred = []
    episode_results: list[EpisodeResult] = []
    window_indices = []

    for ep_id in episode_ids:
        w_idx = episode_to_window[ep_id]
        window_indices.append(w_idx)
        sample = dataset[w_idx]
        obs_dict = sample_to_policy_obs(sample, n_obs, device)
        gt = ground_truth_chunk(sample, n_obs, n_action)

        preds = np.stack(
            [predict_executed_chunk(policy, obs_dict, n_action) for _ in range(args.num_samples)],
            axis=0,
        )  # S, T, D

        all_gt.append(gt)
        all_pred.append(preds)

        mae_per_sample = [float(np.mean(np.abs(p - gt))) for p in preds]
        var_per_dim = preds.var(axis=0).mean(axis=0)
        mae_mean = float(np.mean(mae_per_sample))
        sampling_var_mean = float(var_per_dim.mean())

        episode_results.append(
            EpisodeResult(
                episode_id=int(ep_id),
                window_idx=int(w_idx),
                mae_per_sample=mae_per_sample,
                mae_mean=mae_mean,
                sampling_var_mean=sampling_var_mean,
                sampling_var_per_dim=var_per_dim.tolist(),
            )
        )

        print(
            f"[episode {ep_id:03d}] window={w_idx}  "
            f"MAE={mae_mean:.6f}  sampling_var={sampling_var_mean:.6e}"
        )
        plot_episode(
            out_dir / f"episode_{ep_id:03d}_initial.png",
            ep_id,
            w_idx,
            gt,
            preds,
            ACTION_LABELS,
        )

    gt_arr = np.stack(all_gt, axis=0)          # E, T, D
    pred_arr = np.stack(all_pred, axis=0)      # E, S, T, D
    pred_mean = pred_arr.mean(axis=1)          # E, T, D

    abs_err = np.abs(pred_mean - gt_arr)
    mae_per_dim = abs_err.mean(axis=(0, 1))
    sampling_var_per_dim = pred_arr.var(axis=1).mean(axis=(0, 1))

    np.savez_compressed(
        out_dir / "action_chunks.npz",
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        window_indices=np.asarray(window_indices, dtype=np.int32),
        ground_truth=gt_arr,
        predictions=pred_arr,
        prediction_mean=pred_mean,
        mae_per_dim=mae_per_dim,
        sampling_var_per_dim=sampling_var_per_dim,
        action_labels=np.asarray(ACTION_LABELS),
    )

    with (out_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_id",
                "window_idx",
                "mae_mean",
                "sampling_var_mean",
                *[f"mae_sample_{i}" for i in range(args.num_samples)],
            ]
        )
        for er in episode_results:
            writer.writerow(
                [er.episode_id, er.window_idx, er.mae_mean, er.sampling_var_mean, *er.mae_per_sample]
            )

    summary = {
        "timestamp": timestamp,
        "ckpt": str(ckpt_path),
        "zarr": str(zarr_path),
        "mode": "initial_window_per_episode",
        "episode_ids": episode_ids,
        "window_indices": window_indices,
        "num_episodes": len(episode_ids),
        "num_samples": args.num_samples,
        "n_obs_steps": n_obs,
        "n_action_steps": n_action,
        "overall_mae": float(abs_err.mean()),
        "overall_sampling_var": float(pred_arr.var(axis=1).mean()),
        "mae_per_dim": {k: float(v) for k, v in zip(ACTION_LABELS, mae_per_dim)},
        "sampling_var_per_dim": {k: float(v) for k, v in zip(ACTION_LABELS, sampling_var_per_dim)},
        "episodes": [asdict(e) for e in episode_results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_summary(
        out_dir / "summary_mae_variance.png",
        mae_per_dim,
        sampling_var_per_dim,
        ACTION_LABELS,
        n_episodes=len(episode_ids),
        n_samples=args.num_samples,
    )

    print("\n=== Initial-window action chunk analysis ===")
    print(f"overall MAE (mean pred vs GT): {summary['overall_mae']:.6f}")
    print(f"overall sampling variance:     {summary['overall_sampling_var']:.6e}")
    for er in episode_results:
        print(
            f"  episode {er.episode_id:03d}: MAE={er.mae_mean:.6f}, "
            f"sampling_var={er.sampling_var_mean:.6e}"
        )
    print(f"Wrote {out_dir / 'action_chunks.npz'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary_mae_variance.png'}")
    print(f"Wrote {len(episode_ids)} episode plots to {out_dir}")


if __name__ == "__main__":
    main()
