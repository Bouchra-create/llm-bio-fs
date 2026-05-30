
import argparse
import os
import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.optimizer_factory import run_optimizer
from src.plots import plot_convergence, plot_pareto

from src.evaluation import evaluate_archive
from src.optimizers.stability import run_stability
from src.compare_convergence import compare_optimizers


def main():

    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument("--dataset", required=True,
                        choices=["parkinson", "cancer"])

    # MODEL
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--prompt", default="generic_ml")

    # FEATURE INIT
    parser.add_argument("--init", default="random",
                        choices=["random", "llm", "filter"])

    parser.add_argument("--filter", default="mutual_info",
                        choices=["mutual_info", "anova", "chi2"])

    # OPTIMIZER
    parser.add_argument("--optimizer", default="pso",
                        choices=["pso", "ga", "aco", "woa", "bhho"])

    # EXPERIMENT
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--out", default="results")

    # EXTRA MODES
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--compare_all", action="store_true")

    args = parser.parse_args()

    # LOAD DATA
    Xtr, Xte, ytr, yte, features = load_dataset(args.dataset)

    Xtr = np.array(Xtr)
    Xte = np.array(Xte)
    ytr = np.array(ytr).ravel()
    yte = np.array(yte).ravel()

    tag = f"{args.dataset}_{args.optimizer}_{args.init}_{args.filter}_{args.model.replace(':','_')}_{args.prompt}"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT:", tag)
    print("=" * 60)

    # ======================================================
    # STABILITY MODE (10 RUNS)
    # ======================================================
    if args.stability:

        summary, results, curves = run_stability(
            args.optimizer,
            Xtr, Xte, ytr, yte,
            features,
            args,
            n_runs=10
        )

        print("\n===== STABILITY =====")
        print(f"F1  : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        print(f"HV  : {summary['hv_mean']:.4f} ± {summary['hv_std']:.4f}")
        print(f"GD+ : {summary['gd_mean']:.4f} ± {summary['gd_std']:.4f}")

        pd.DataFrame([summary]).to_csv(
            os.path.join(out_dir, "stability.csv"),
            index=False
        )

    # ======================================================
    # SINGLE RUN
    # ======================================================
    else:

        archive, curve, test_f1, hv, selected = run_optimizer(
            args.optimizer,
            X=Xtr,
            Xte=Xte,
            y=ytr,
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

        print("\n===== RESULTS =====")
        print("F1 :", round(test_f1, 4))
        print("HV :", round(metrics["hv"], 4))
        print("GD+:", metrics["gd_plus"])
        print("UNFR:", metrics["unfr"])
        print("Delta:", metrics["delta"])
        print("Selected:", len(selected))

        # SAVE PLOTS
        plot_convergence(curve,
                         f"Convergence-{tag}",
                         os.path.join(out_dir, "convergence.png"))

        plot_pareto(archive,
                    f"Pareto-{tag}",
                    os.path.join(out_dir, "pareto.png"))

        # SAVE CSV
        result = pd.DataFrame([{
            "dataset": args.dataset,
            "optimizer": args.optimizer,
            "init": args.init,
            "filter": args.filter,
            "model": args.model,
            "prompt": args.prompt,
            "f1": test_f1,
            "hv": metrics["hv"],
            "gd_plus": metrics["gd_plus"],
            "unfr": metrics["unfr"],
            "delta": metrics["delta"],
            "selected": len(selected)
        }])

        csv_path = os.path.join(out_dir, "results.csv")

        if os.path.exists(csv_path):
            old = pd.read_csv(csv_path)
            result = pd.concat([old, result], ignore_index=True)

        result.to_csv(csv_path, index=False)

        print("\nSaved:", csv_path)

    # ======================================================
    # COMPARE ALL OPTIMIZERS
    # ======================================================
    if args.compare_all:

        optimizers = ["PSO", "GA", "ACO", "WOA", "BHHO"]

        curves = compare_optimizers(
            optimizers,
            Xtr, Xte, ytr, yte,
            features,
            args
        )

        print("\nComparison completed.")


if __name__ == "__main__":
    main()