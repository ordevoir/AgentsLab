#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize TorchRL CSV logs (train & eval) with matplotlib.

Assumes run directory structure:
runs/<algo>_<env>_<YYYYMMDD-HHMMSS>/
  csv_logs/{train,eval}/*.csv
  meta.yaml (optional)

Usage examples:
  python visualize_logs.py --root ..                      # latest run
  python visualize_logs.py --run-path ../runs/ppo_pendulum_20250101-120000
  python visualize_logs.py --root .. --smoothing 21       # moving average window
  python visualize_logs.py --save-only                    # do not open GUI window
"""

import argparse
from pathlib import Path
import re
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml  # optional, for meta.yaml
except Exception:
    yaml = None

# ------------------------------- utils ---------------------------------------

RUN_NAME_RE = re.compile(r"^(.+?)_(.+?)_(\d{8}-\d{6})$")

def find_latest_run(root: Path) -> Path:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    candidates = []
    for d in runs_dir.iterdir():
        if d.is_dir() and RUN_NAME_RE.match(d.name):
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No run directories matching pattern in {runs_dir}")
    # sort by timestamp embedded in name; fallback to mtime
    def keyfun(p: Path):
        m = RUN_NAME_RE.match(p.name)
        return (m.group(3) if m else "", p.stat().st_mtime)
    return sorted(candidates, key=keyfun)[-1]

def read_meta(run_path: Path) -> dict:
    meta_path = run_path / "meta.yaml"
    if yaml is None or not meta_path.exists():
        return {}
    try:
        return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def load_csv_dir(csv_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV files from a directory into a single wide DataFrame.
    Supports both 'long' (step,name,value) and 'wide' (step, metric1, metric2, ...) formats.
    """
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")
    dfs = []
    for f in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue

        # normalize column names
        df.columns = [str(c).strip() for c in df.columns]

        if {"step", "name", "value"}.issubset(df.columns):
            # long -> wide
            pivot = df.pivot_table(index="step", columns="name", values="value", aggfunc="last").reset_index()
            dfs.append(pivot)
        elif "step" in df.columns:
            # already wide
            dfs.append(df)
        else:
            # try to guess a step column
            step_col = None
            for c in df.columns:
                if c.lower() in {"global_frames", "frames", "t", "iteration"}:
                    step_col = c
                    break
            if step_col is None:
                print(f"[WARN] Can't find 'step' column in {f}, skipping", file=sys.stderr)
                continue
            df = df.rename(columns={step_col: "step"})
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No readable CSV files in {csv_dir}")

    # merge on step, prefer last occurrence for duplicate columns
    out = dfs[0]
    for df in dfs[1:]:
        out = pd.merge(out, df, on="step", how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in out.columns if c.endswith("_dup")]
        if dup_cols:
            out = out.drop(columns=dup_cols)
    out = out.sort_values("step").reset_index(drop=True)
    # drop non-numeric metric columns except 'step'
    for c in list(out.columns):
        if c == "step":
            continue
        # coerce to numeric where possible
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def moving_average(x: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return x
    return x.rolling(window=window, min_periods=max(1, window // 3), center=False).mean()

def ensure_fig_dir(run_path: Path) -> Path:
    fig_dir = run_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

def pick_present_metrics(df: pd.DataFrame, candidates):
    return [m for m in candidates if m in df.columns]

def plot_metric(df: pd.DataFrame, metric: str, title: str, fig_path: Path,
                smoothing: int = 1, xlabel: str = "step"):
    y = df[metric].copy()
    x = df["step"].copy()
    y_sm = moving_average(y, smoothing) if smoothing and smoothing > 1 else y

    plt.figure()
    plt.plot(x, y_sm)
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    # Show only in interactive mode; caller decides to plt.show() or not.

# ------------------------------- main ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize TorchRL CSV logs (train & eval) with matplotlib.")
    ap.add_argument("--root", type=Path, default=Path("..").resolve(), help="Project ROOT containing 'runs/'")
    ap.add_argument("--run-path", type=Path, default=None, help="Path to a specific run folder (overrides --root)")
    ap.add_argument("--smoothing", type=int, default=11, help="Moving average window (odd integer recommended). 1=off")
    ap.add_argument("--save-only", action="store_true", help="Save figures but do not open GUI window")
    ap.add_argument("--train-metrics", type=str, nargs="*", default=None,
                    help="Which train metrics to plot (default: auto-detect common ones)")
    ap.add_argument("--eval-metrics", type=str, nargs="*", default=None,
                    help="Which eval metrics to plot (default: auto-detect common ones)")
    args = ap.parse_args()

    if args.run_path is None:
        run_path = find_latest_run(args.root)
        print(f"[INFO] Using latest run: {run_path}")
    else:
        run_path = args.run_path
        print(f"[INFO] Using run: {run_path}")

    meta = read_meta(run_path)
    algo = meta.get("algo", None)
    env = meta.get("env", None)
    run_name = run_path.name

    # Load CSV logs
    train_dir = run_path / "csv_logs" / "train"
    eval_dir  = run_path / "csv_logs" / "eval"
    train_df = load_csv_dir(train_dir)
    eval_df  = load_csv_dir(eval_dir)

    # Default metric sets (filter by presence)
    default_train_candidates = [
        "reward/avg",
        "loss/objective", "loss/critic", "loss/entropy",
        "lr",
    ]
    default_eval_candidates = [
        "return_mean",
        "max_episode_length",
    ]
    train_metrics = args.train_metrics or pick_present_metrics(train_df, default_train_candidates)
    eval_metrics  = args.eval_metrics  or pick_present_metrics(eval_df,  default_eval_candidates)

    if not train_metrics:
        # fall back: plot all numeric columns except 'step'
        train_metrics = [c for c in train_df.columns if c != "step"]
    if not eval_metrics:
        eval_metrics = [c for c in eval_df.columns if c != "step"]

    fig_dir = ensure_fig_dir(run_path)

    subtitle = f"{run_name}"
    if algo and env:
        subtitle = f"{algo} on {env} — {run_name}"

    # Plot train
    for m in train_metrics:
        title = f"Train: {m} ({subtitle})"
        out = fig_dir / f"train_{m.replace('/', '_')}.png"
        plot_metric(train_df, m, title, out, smoothing=args.smoothing)

    # Plot eval
    for m in eval_metrics:
        title = f"Eval: {m} ({subtitle})"
        out = fig_dir / f"eval_{m.replace('/', '_')}.png"
        plot_metric(eval_df, m, title, out, smoothing=args.smoothing)

    print(f"[INFO] Saved figures to: {fig_dir}")

    if not args.save_only:
        # Show one by one in interactive sessions
        plt.show()

if __name__ == "__main__":
    main()
