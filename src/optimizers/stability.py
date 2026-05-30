import numpy as np
import pandas as pd

from src.optimizer_factory import run_optimizer
from src.evaluation import evaluate_archive


def run_stability(
    optimizer_name,
    X, Xte, y, yte, features,
    args,
    n_runs=10
):

    results = {
        "f1": [],
        "hv": [],
        "gd": []
    }

    all_curves = []

    for seed in range(n_runs):

        np.random.seed(seed)

        archive, curve, f1, hv, selected = run_optimizer(
            optimizer_name,
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

        metrics = evaluate_archive(archive)

        results["f1"].append(f1)
        results["hv"].append(metrics["hv"])
        results["gd"].append(metrics["gd_plus"])

        all_curves.append(curve)

    summary = {
        "f1_mean": np.mean(results["f1"]),
        "f1_std": np.std(results["f1"]),

        "hv_mean": np.mean(results["hv"]),
        "hv_std": np.std(results["hv"]),

        "gd_mean": np.mean(results["gd"]),
        "gd_std": np.std(results["gd"])
    }

    return summary, results, all_curves