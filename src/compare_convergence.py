import numpy as np
import matplotlib.pyplot as plt

from src.optimizer_factory import run_optimizer


def compare_optimizers(
    optimizers,
    X, Xte, y, yte, features,
    args
):

    curves = {}

    for opt in optimizers:

        _, curve, _, _, _ = run_optimizer(
            opt,
            X=X,
            Xte=Xte,
            y=y,
            yte=yte,
            features=features,
            init=args.init,
            model=args.model,
            prompt=args.prompt,
            filter_type=args.filter,
            pop=args.pop,
            iters=args.iters
        )

        curves[opt] = curve

    # ================= PLOT =================
    plt.figure()

    for k, v in curves.items():
        plt.plot(v, label=k)

    plt.xlabel("Iterations")
    plt.ylabel("Best F1")
    plt.title("Convergence Comparison (All Optimizers)")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/convergence_comparison.png", dpi=300)
    plt.close()

    return curves