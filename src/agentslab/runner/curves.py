from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_from_csv(log_dir: str):
    """Find CSV files in `log_dir` and draw a few standard curves."""
    csv_path = os.path.join(log_dir, "train.csv")
    if not os.path.exists(csv_path):
        # TorchRL CSVLogger writes <exp_name>.csv by default
        # so try dqn.csv / train.csv fallbacks.
        for cand in ["dqn.csv", "train.csv"]:
            cand_path = os.path.join(log_dir, cand)
            if os.path.exists(cand_path):
                csv_path = cand_path
                break
    df = pd.read_csv(csv_path)
    # try common columns
    cols = [c for c in df.columns if any(x in c.lower() for x in ["reward", "loss", "eps", "epsilon"])]
    if not cols:
        cols = df.columns[:4]
    for c in cols:
        plt.figure()
        plt.plot(df[c])
        plt.title(c)
        plt.xlabel("step")
        plt.ylabel(c)
        plt.grid(True)
    plt.show()
