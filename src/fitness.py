import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

SEED = 42


def fitness(mask, X, y, n_splits=5):

    mask = np.array(mask)

    # ========================================================
    # INVALID SOLUTION HANDLING
    # ========================================================
    if np.sum(mask) == 0:
        return [1e6, 1e6]

    X_sel = X[:, mask == 1]

    # safety: avoid empty or degenerate feature matrices
    if X_sel.shape[1] == 0:
        return [1e6, 1e6]

    # ========================================================
    # CROSS VALIDATION
    # ========================================================
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1
    )

    scores = []

    for tr, te in skf.split(X_sel, y):

        X_train, X_test = X_sel[tr], X_sel[te]
        y_train, y_test = y[tr], y[te]

        # safety check
        if X_train.shape[1] == 0:
            continue

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        scores.append(
            f1_score(y_test, pred, average="weighted")
        )

    # fallback if something went wrong
    if len(scores) == 0:
        return [1e6, 1e6]

    # ========================================================
    # OBJECTIVES (multi-objective FS)
    # ========================================================

    f1 = np.mean(scores)

    feature_ratio = np.sum(mask) / len(mask)

    # optional: penalty for very large subsets
    sparsity_penalty = feature_ratio ** 2

    return [
        feature_ratio,        # minimize
        -f1,                  # minimize (-F1)
    ]