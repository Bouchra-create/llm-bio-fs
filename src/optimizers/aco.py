import numpy as np
from src.fitness import fitness
from src.utils import update_archive
from src.evaluation import evaluate_archive


def build_solution(pheromone):
    prob = pheromone
    mask = (np.random.rand(len(prob)) < prob).astype(int)

    if np.sum(mask) == 0:
        mask[np.argmax(prob)] = 1

    return mask


def run_aco(X, Xte, y, yte, features,init="random",model="llama3.2:3b",prompt="generic_ml",filter_type="mutual_info", pop=20, iters=30):

    dim = X.shape[1]
    pheromone = np.ones(dim) * 0.5

    archive = []
    curve = []

    for it in range(iters):

        ants = [build_solution(pheromone) for _ in range(pop)]
        fits = [fitness(a, X, y) for a in ants]

        for i in range(pop):
            archive = update_archive(archive, ants[i], fits[i])

        # evaporation
        pheromone *= 0.9

        # reinforcement
        for i in range(pop):
            f1 = -fits[i][1]
            pheromone += ants[i] * f1

        pheromone = np.clip(pheromone, 0.01, 1.0)

        best_f1 = -min([a["f"][1] for a in archive])
        curve.append(best_f1)

        print(f"[ACO] Iter {it+1} | Best F1={best_f1:.4f}")

    # ========================================================
    # FINAL EVALUATION (CORRECT PLACE)
    # ========================================================
    metrics = evaluate_archive(archive)

    best = max(archive, key=lambda x: -x["f"][1])
    mask = best["p"]

    return (
        archive,
        curve,
        best_f1,
        metrics,   # <<< IMPORTANT FIX (ALL METRICS RETURNED)
        features[mask == 1]
    )