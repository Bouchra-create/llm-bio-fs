import numpy as np
from src.fitness import fitness
from src.metrics import hypervolume
from src.utils import update_archive

SEED = 42
np.random.seed(SEED)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def binarize(x):
    return (np.random.rand(len(x)) < sigmoid(x)).astype(int)


def run_woa(X, Xte, y, yte, features,init="random",model="llama3.2:3b",prompt="generic_ml",filter_type="mutual_info", pop=20, iters=30):

    dim = X.shape[1]
    pos = np.random.uniform(-1, 1, (pop, dim))

    archive = []
    curve = []

    for it in range(iters):

        for i in range(pop):

            mask = binarize(pos[i])
            f = fitness(mask, X, y)
            archive = update_archive(archive, mask, f)

        best = max(archive, key=lambda x: -x["f"][1])["p"]

        a = 2 - it * (2 / iters)

        for i in range(pop):

            r = np.random.rand(dim)
            A = 2 * a * r - a
            C = 2 * r

            if np.random.rand() < 0.5:
                D = np.abs(C * best - pos[i])
                pos[i] = best - A * D
            else:
                rand = pos[np.random.randint(pop)]
                D = np.abs(C * rand - pos[i])
                pos[i] = rand - A * D

        best_f1 = -min([a["f"][1] for a in archive])
        curve.append(best_f1)

        print(f"[WOA] Iter {it+1} | Best F1={best_f1:.4f}")

    best = max(archive, key=lambda x: -x["f"][1])
    mask = best["p"]

    return archive, curve, best_f1, hypervolume([a["f"] for a in archive]), features[mask == 1]