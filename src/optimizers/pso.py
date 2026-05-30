import numpy as np
from src.fitness import fitness
from src.metrics import hypervolume
from src.utils import update_archive, dominates, sigmoid
from src.initializers import get_init_scores

SEED = 42
np.random.seed(SEED)


def init_population(pop, dim):
    pos = np.random.randint(0, 2, (pop, dim))
    vel = np.random.uniform(-1, 1, (pop, dim))
    return pos, vel


def init_llm_population(pop, dim, scores):

    P = []

    for i in range(pop):

        if i < pop // 3:
            threshold = np.percentile(scores, 85)
            p = (scores >= threshold).astype(int)

        elif i < 2 * pop // 3:
            p = (np.random.rand(dim) < scores * 0.3).astype(int)

        else:
            p = (np.random.rand(dim) < 0.1).astype(int)

        if np.sum(p) == 0:
            p[np.argmax(scores)] = 1

        P.append(p)

    vel = np.random.uniform(-1, 1, (pop, dim))
    return np.array(P), vel


def run_pso(
    X, Xte, y, yte, features,
    init="random",
    model=None,
    prompt=None,
    filter_type="mutual_info",
    pop=20,
    iters=30
):

    dim = X.shape[1]

    scores = get_init_scores(
        init, X, y, features,
        model=model,
        prompt=prompt,
        filter_type=filter_type
    )

    if init in ["llm", "filter"]:
        pos, vel = init_llm_population(pop, dim, scores)
    else:
        pos, vel = init_population(pop, dim)

    pbest = pos.copy()
    pbest_fit = [fitness(p, X, y) for p in pos]

    archive = []
    curve = []

    for i in range(pop):
        archive = update_archive(archive, pos[i], pbest_fit[i])

    for it in range(iters):

        for i in range(pop):

            f = fitness(pos[i], X, y)

            if dominates(f, pbest_fit[i]):
                pbest[i] = pos[i].copy()
                pbest_fit[i] = f

            archive = update_archive(archive, pos[i], f)

        best_f1 = -min([a["f"][1] for a in archive])
        curve.append(best_f1)

        for i in range(pop):

            leader = archive[np.random.randint(len(archive))]["p"]

            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            vel[i] = (
                0.7 * vel[i]
                + 2 * r1 * (pbest[i] - pos[i])
                + 2 * r2 * (leader - pos[i])
            )

            vel[i] = np.clip(vel[i], -4, 4)

            pos[i] = (np.random.rand(dim) < sigmoid(vel[i])).astype(int)

            if np.sum(pos[i]) == 0:
                pos[i][np.random.randint(dim)] = 1

        print(f"[PSO] Iter {it+1} | Best F1={best_f1:.4f}")

    best = max(archive, key=lambda x: -x["f"][1])
    mask = best["p"]

    return archive, curve, best_f1, hypervolume([a["f"] for a in archive]), features[mask == 1]