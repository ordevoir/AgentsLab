import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_training_curves(csv_path: str, out_png: Optional[str] = None):
    df = pd.read_csv(csv_path)
    # Three plots: train_reward, eval_return, lr
    plt.figure()
    df.plot(x="iter", y="train_reward_mean")
    plt.title("Train: mean reward per batch")
    plt.xlabel("iteration")
    plt.ylabel("reward")
    if out_png:
        plt.savefig(out_png, bbox_inches="tight")

    plt.figure()
    if "eval_return" in df.columns:
        df.plot(x="iter", y="eval_return")
        plt.title("Eval: episode return")
        plt.xlabel("iteration")
        plt.ylabel("return")
        if out_png:
            stem = out_png.rsplit(".", 1)[0]
            plt.savefig(stem + "_eval.png", bbox_inches="tight")

    plt.figure()
    if "lr" in df.columns:
        df.plot(x="iter", y="lr")
        plt.title("Learning rate")
        plt.xlabel("iteration")
        plt.ylabel("lr")
        if out_png:
            stem = out_png.rsplit(".", 1)[0]
            plt.savefig(stem + "_lr.png", bbox_inches="tight")
