import numpy as np
from src.fitness import fitness
from src.metrics import hypervolume

SEED = 42
np.random.seed(SEED)


def tournament_selection(pop, fits, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[min(idx, key=lambda i: fits[i][1])]


def crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2)


def mutate(ind, rate=0.02):
    m = np.random.rand(len(ind)) < rate
    ind[m] = 1 - ind[m]
    if np.sum(ind) == 0:
        ind[np.random.randint(len(ind))] = 1
    return ind


def run_ga(X, Xte, y, yte, features,init="random",model="llama3.2:3b",prompt="generic_ml",filter_type="mutual_info", pop=20, iters=30):

    dim = X.shape[1]
    popu = np.random.randint(0, 2, (pop, dim))

    archive = []
    curve = []

    for it in range(iters):

        fits = [fitness(ind, X, y) for ind in popu]

        from src.utils import update_archive
        for i in range(pop):
            archive = update_archive(archive, popu[i], fits[i])

        best_f1 = -min([a["f"][1] for a in archive])
        curve.append(best_f1)

        new_pop = []

        for _ in range(pop):
            p1 = tournament_selection(popu, fits)
            p2 = tournament_selection(popu, fits)

            child = mutate(crossover(p1, p2))
            new_pop.append(child)

        popu = np.array(new_pop)

    best = max(archive, key=lambda x: -x["f"][1])
    mask = best["p"]

    return archive, curve, best_f1, hypervolume([a["f"] for a in archive]), features[mask == 1]