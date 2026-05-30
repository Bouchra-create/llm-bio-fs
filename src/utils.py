import numpy as np

SEED = 42

np.random.seed(SEED)


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def update_archive(archive, p, f):

    new_archive = []
    for sol in archive:

        if dominates(sol["f"], f):
            return archive

        if not dominates(f, sol["f"]):
            new_archive.append(sol)

    new_archive.append({"p": p.copy(), "f": f})

    return new_archive

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hypervolume(front):

    front = np.array(front)

    front = front[
        front[:, 0].argsort()
    ]

    ref = [1.1, 0]

    hv = 0

    prev = ref[1]

    for f1, f2 in front:

        w = ref[0] - f1

        h = prev - f2

        if w > 0 and h > 0:

            hv += w * h

        prev = f2

    return hv