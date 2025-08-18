from __future__ import annotations
import os, csv
import matplotlib.pyplot as plt

def plot_training(log_csv_path: str):
    frames, losses, eps = [], [], []
    if not os.path.exists(log_csv_path):
        print(f"No train log at: {log_csv_path}")
        return
    with open(log_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(float(row["frames"]))
            losses.append(float(row["loss"]))
            v = row.get("eps", "")
            try:
                eps.append(float(v))
            except Exception:
                eps.append(None)
    if not frames:
        print("Train log is empty.")
        return
    plt.figure()
    plt.plot(frames, losses)
    plt.xlabel("Frames")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Frames")
    plt.show()
    if any(v is not None for v in eps):
        xs = [f for f, e in zip(frames, eps) if e is not None]
        ys = [e for e in eps if e is not None]
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("Frames")
        plt.ylabel("Epsilon")
        plt.title("Exploration Epsilon vs Frames")
        plt.show()

def plot_eval(log_csv_path: str):
    eps, rets = [], []
    if not os.path.exists(log_csv_path):
        print(f"No eval log at: {log_csv_path}")
        return
    with open(log_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(int(row["episode"]))
            rets.append(float(row["return"]))
    if not eps:
        print("Eval log is empty.")
        return
    plt.figure()
    plt.plot(eps, rets)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Evaluation Returns")
    plt.show()
