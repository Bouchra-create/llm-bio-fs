import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(curve, title, save_path):

    plt.figure(figsize=(8, 5))

    plt.plot(curve)

    plt.xlabel("Iteration")
    plt.ylabel("Best F1")
    plt.title(title)

    plt.grid(True)

    # FIX HERE
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_pareto(archive, title, save_path):

    front = np.array([a["f"] for a in archive])

    plt.figure(figsize=(6, 6))

    plt.scatter(
        front[:, 0],
        -front[:, 1]
    )

    plt.xlabel("Feature Ratio")
    plt.ylabel("F1 Score")

    plt.title(title)

    plt.grid(True)

    # FIX HERE
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()