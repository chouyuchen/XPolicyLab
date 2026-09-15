#!/usr/bin/env python3
"""Plot DP train / val loss from a Hydra run's logs.json.txt.

Examples
--------
# one finished run directory
python plot_train_curves.py \\
  data/outputs/2026.08.19/09.14.25_robot_skill_open_cabinet_drawer_skill_open_cabinet_drawer

# newest run under data/outputs
python plot_train_curves.py --latest

# every run that has logs.json.txt
python plot_train_curves.py --all

# overlay several runs
python plot_train_curves.py --all --compare

# custom log file
python plot_train_curves.py path/to/logs.json.txt --out-dir ./my_plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Activate your DP env, e.g.\n"
        "  conda activate xpl_dp && pip install matplotlib"
    ) from e


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = SCRIPT_DIR / "data" / "outputs"


def find_log_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    for name in ("logs.json.txt", "logs.json", "log.json.txt"):
        cand = path / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No logs.json.txt under {path}")


def discover_runs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    return [p.parent for p in sorted(root.rglob("logs.json.txt"))]


def parse_logs(log_path: Path) -> dict:
    train_by_epoch: dict[int, list[float]] = defaultdict(list)
    val_rows: list[dict] = []
    n_lines = 0

    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_lines += 1
            if "epoch" not in row:
                continue
            ep = int(row["epoch"])
            if "train_loss" in row:
                train_by_epoch[ep].append(float(row["train_loss"]))
            if "val_loss" in row:
                val_rows.append(
                    {
                        "epoch": ep,
                        "val_loss": float(row["val_loss"]),
                        "global_step": int(row.get("global_step", -1)),
                    }
                )

    if not train_by_epoch:
        raise ValueError(f"No train_loss entries in {log_path}")

    epochs = sorted(train_by_epoch)
    train_mean = np.array(
        [float(np.mean(train_by_epoch[e])) for e in epochs], dtype=float
    )

    val_by_ep: dict[int, float] = {}
    for r in val_rows:
        val_by_ep[r["epoch"]] = r["val_loss"]
    val_epochs = sorted(val_by_ep)
    val_loss = np.array([val_by_ep[e] for e in val_epochs], dtype=float)

    best_val = None
    if len(val_epochs):
        i = int(np.argmin(val_loss))
        best_val = {"epoch": int(val_epochs[i]), "val_loss": float(val_loss[i])}

    return {
        "log_path": log_path,
        "run_dir": log_path.parent,
        "run_name": log_path.parent.name,
        "n_lines": n_lines,
        "epochs": np.array(epochs, dtype=int),
        "train_mean": train_mean,
        "val_epochs": np.array(val_epochs, dtype=int),
        "val_loss": val_loss,
        "best_val": best_val,
    }


def write_csv(stats: dict, csv_path: Path) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "split", "loss"])
        for e, v in zip(stats["epochs"], stats["train_mean"]):
            w.writerow([int(e), "train", f"{float(v):.8f}"])
        for e, v in zip(stats["val_epochs"], stats["val_loss"]):
            w.writerow([int(e), "val", f"{float(v):.8f}"])


def short_name(run_name: str) -> str:
    name = run_name
    for prefix in ("robot_",):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    # drop duplicated skill suffix if present: skill_x_skill_x
    parts = name.split("_")
    if len(parts) >= 4 and "_".join(parts[: len(parts) // 2]) == "_".join(
        parts[len(parts) // 2 :]
    ):
        name = "_".join(parts[: len(parts) // 2])
    return name


def plot_single(stats: dict, png_path: Path) -> None:
    title = short_name(stats["run_name"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)

    ax = axes[0]
    ax.plot(
        stats["epochs"],
        stats["train_mean"],
        color="#2563eb",
        lw=1.6,
        label="train_loss (epoch mean)",
    )
    if len(stats["val_epochs"]):
        ax.plot(
            stats["val_epochs"],
            stats["val_loss"],
            "o-",
            color="#dc2626",
            lw=1.6,
            ms=4,
            label="val_loss",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion loss")
    ax.set_title(f"{title} — linear")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        stats["epochs"],
        stats["train_mean"],
        color="#2563eb",
        lw=1.6,
        label="train_loss",
    )
    if len(stats["val_epochs"]):
        ax.plot(
            stats["val_epochs"],
            stats["val_loss"],
            "o-",
            color="#dc2626",
            lw=1.6,
            ms=4,
            label="val_loss",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion loss (log)")
    ax.set_title(f"{title} — log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def plot_compare(all_stats: list[dict], png_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    for stats in all_stats:
        label = short_name(stats["run_name"])
        axes[0].plot(
            stats["epochs"], stats["train_mean"], lw=1.4, label=f"{label} train"
        )
        if len(stats["val_epochs"]):
            axes[1].plot(
                stats["val_epochs"],
                stats["val_loss"],
                "o-",
                lw=1.4,
                ms=3,
                label=f"{label} val",
            )

    axes[0].set_title("Train loss (epoch mean)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Diffusion loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Val loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Diffusion loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def print_summary(stats: dict) -> None:
    e0 = int(stats["epochs"][0])
    eN = int(stats["epochs"][-1])
    t0 = float(stats["train_mean"][0])
    tN = float(stats["train_mean"][-1])
    print(f"\n=== {stats['run_name']} ===")
    print(f"log: {stats['log_path']}")
    print(f"steps logged: {stats['n_lines']}")
    print(f"train_loss: {t0:.6f} (ep{e0}) → {tN:.6f} (ep{eN})")
    if len(stats["val_epochs"]):
        v0 = float(stats["val_loss"][0])
        vN = float(stats["val_loss"][-1])
        print(
            f"val_loss:   {v0:.6f} (ep{int(stats['val_epochs'][0])}) → "
            f"{vN:.6f} (ep{int(stats['val_epochs'][-1])})"
        )
        b = stats["best_val"]
        print(f"best val:   {b['val_loss']:.6f} @ epoch {b['epoch']}")
    else:
        print("val_loss:   (none logged)")


def resolve_targets(args: argparse.Namespace) -> list[Path]:
    if args.all or args.latest or (args.compare and not args.targets):
        runs = discover_runs(Path(args.outputs_root))
        if not runs:
            raise SystemExit(f"No logs.json.txt under {args.outputs_root}")
        if args.latest and not args.all:
            runs = sorted(
                runs, key=lambda p: (p / "logs.json.txt").stat().st_mtime
            )
            return [runs[-1]]
        return runs

    if not args.targets:
        raise SystemExit(
            "Pass a run dir / logs.json.txt, or use --all / --latest"
        )

    return [find_log_file(Path(t)).parent for t in args.targets]


def process_run(run_dir: Path, out_dir: Path | None) -> dict:
    log_path = find_log_file(run_dir)
    stats = parse_logs(log_path)
    dest = out_dir if out_dir is not None else stats["run_dir"]
    dest.mkdir(parents=True, exist_ok=True)

    # If dumping many runs into one --out-dir, avoid overwriting.
    if out_dir is not None:
        stem = short_name(stats["run_name"])
        csv_path = dest / f"{stem}_loss_curves.csv"
        png_path = dest / f"{stem}_loss_curves.png"
    else:
        csv_path = dest / "loss_curves.csv"
        png_path = dest / "loss_curves.png"

    write_csv(stats, csv_path)
    plot_single(stats, png_path)
    print_summary(stats)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot DP train_loss / val_loss curves from logs.json.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "targets",
        nargs="*",
        help="Run directory and/or logs.json.txt path",
    )
    p.add_argument(
        "--outputs-root",
        default=str(DEFAULT_OUTPUTS),
        help=f"Root used by --all/--latest (default: {DEFAULT_OUTPUTS})",
    )
    p.add_argument(
        "--all", action="store_true", help="Process every run under outputs-root"
    )
    p.add_argument(
        "--latest", action="store_true", help="Process only the newest run"
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Also write a compare overlay figure",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional directory for png/csv (default: each run dir)",
    )
    p.add_argument(
        "--compare-out",
        default=None,
        help="Compare figure path (default: <outputs-root>/compare_loss_curves.png)",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.compare and not args.targets and not args.all and not args.latest:
        args.all = True

    run_dirs = resolve_targets(args)
    out_dir = (
        Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    )

    all_stats: list[dict] = []
    for run_dir in run_dirs:
        try:
            all_stats.append(process_run(run_dir, out_dir))
        except Exception as e:
            print(f"[skip] {run_dir}: {e}", file=sys.stderr)

    if args.compare and len(all_stats) >= 2:
        compare_path = (
            Path(args.compare_out).expanduser().resolve()
            if args.compare_out
            else Path(args.outputs_root).expanduser().resolve()
            / "compare_loss_curves.png"
        )
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        plot_compare(all_stats, compare_path)
        print(f"\nwrote compare figure: {compare_path}")
    elif args.compare and len(all_stats) < 2:
        print("[warn] --compare needs ≥2 successful runs", file=sys.stderr)

    return 0 if all_stats else 1


if __name__ == "__main__":
    raise SystemExit(main())
