import numpy as np

from sklearn.feature_selection import mutual_info_classif, f_classif, chi2


def compute_filter_scores(X, y, method="mutual_info"):

    X = np.array(X)
    y = np.array(y).ravel()

    if method == "mutual_info":

        scores = mutual_info_classif(X, y, random_state=42)

    elif method == "anova":

        scores = f_classif(X, y)[0]

    elif method == "chi2":

        # IMPORTANT: chi2 requires non-negative values
        if np.any(X < 0):
            X = X - np.min(X)

        scores = chi2(X, y)[0]

    else:
        raise ValueError(
            f"Unknown filter method: {method}. "
            "Use: mutual_info | anova | chi2"
        )

    # =========================
    # SAFE CLEANING
    # =========================
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # =========================
    # NORMALIZATION (robust)
    # =========================
    if np.max(scores) > np.min(scores):
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    else:
        scores = np.zeros_like(scores)

    return scores