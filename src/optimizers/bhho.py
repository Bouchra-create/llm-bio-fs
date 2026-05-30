import numpy as np
from src.fitness import fitness
from src.metrics import hypervolume
from src.utils import update_archive

SEED = 42
np.random.seed(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_bhho(X, Xte, y, yte, features,init="random",model="llama3.2:3b",prompt="generic_ml",filter_type="mutual_info", pop=20, iters=30):

    dim = X.shape[1]
    pos = np.random.uniform(-1, 1, (pop, dim))

    archive = []
    curve = []

    for it in range(iters):

        for i in range(pop):

            mask = (np.random.rand(dim) < sigmoid(pos[i])).astype(int)

            f = fitness(mask, X, y)
            archive = update_archive(archive, mask, f)

        best = max(archive, key=lambda x: -x["f"][1])["p"]

        E1 = 2 * (1 - it / iters)

        for i in range(pop):

            E0 = 2 * np.random.rand(dim) - 1
            escape = E1 * E0

            if np.mean(np.abs(escape)) >= 1:
                pos[i] += np.random.randn(dim)
            else:
                pos[i] = best - np.random.rand(dim) * (best - pos[i])

        best_f1 = -min([a["f"][1] for a in archive])
        curve.append(best_f1)

        print(f"[BHHO] Iter {it+1} | Best F1={best_f1:.4f}")

    best = max(archive, key=lambda x: -x["f"][1])
    mask = best["p"]

    return archive, curve, best_f1, hypervolume([a["f"] for a in archive]), features[mask == 1]