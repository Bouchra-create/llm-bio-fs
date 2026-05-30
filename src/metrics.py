import numpy as np

# ============================================================
# HYPERVOLUME
# ============================================================
def hypervolume(front, ref_point=(1.1, 0.0)):
    front = np.array(front)
    front = front[np.argsort(front[:, 0])]

    hv = 0.0
    prev_f2 = ref_point[1]

    for f1, f2 in front:
        width = ref_point[0] - f1
        height = prev_f2 - f2

        if width > 0 and height > 0:
            hv += width * height

        prev_f2 = min(prev_f2, f2)

    return hv


# ============================================================
# DOMINANCE
# ============================================================
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


# ============================================================
# UNIQUE NONDOMINATED FRONT RATIO (UNFR)
# ============================================================
def unfr(archive, true_front):
    combined = np.array(archive + true_front)

    nondom = []
    for i, a in enumerate(combined):
        if not any(i != j and dominates(combined[j], a)
                   for j in range(len(combined))):
            nondom.append(a)

    return len(nondom) / len(combined)


# ============================================================
# GENERATIONAL DISTANCE + (GD+)
# ============================================================
def gd_plus(archive, true_front):
    A = np.array(archive)
    T = np.array(true_front)

    dists = []
    for a in A:
        d = np.linalg.norm(np.maximum(T - a, 0), axis=1)
        dists.append(np.min(d))

    return np.mean(dists)


# ============================================================
# DELTA SPREAD
# ============================================================
def delta_spread(archive):
    A = np.array(archive)

    if len(A) < 2:
        return 0.0

    A = A[np.argsort(A[:, 0])]

    d = np.linalg.norm(np.diff(A, axis=0), axis=1)
    d_mean = np.mean(d)

    df = np.linalg.norm(A[0] - A[-1])

    num = np.sum(np.abs(d - d_mean))
    den = len(d) * (d_mean + df)

    return num / den if den != 0 else 0