from __future__ import annotations
from matplotlib import pyplot as plt

def plot_series(values, title: str, xlabel="Step", ylabel="Value"):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
