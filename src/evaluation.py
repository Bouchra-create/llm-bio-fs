import numpy as np
from src.metrics import hypervolume, unfr, gd_plus, delta_spread


def evaluate_archive(archive, true_front=None):

    front = np.array([a["f"] for a in archive])

    hv = hypervolume(front)
    delta = delta_spread(front)

    # fallback = self-reference (NOT for papers, but stable)
    if true_front is None:
        gd = 0.0
        unfr_value = 1.0   # all considered nondominated baseline
    else:
        gd = gd_plus(front, np.array(true_front))
        unfr_value = unfr(front, true_front)

    return {
        "hv": hv,
        "gd_plus": gd,
        "unfr": unfr_value,
        "delta": delta
    }