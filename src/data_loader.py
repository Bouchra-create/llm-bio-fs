import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

SEED = 42


def load_dataset(name):

    if name == "parkinson":

        train_path = "data/parkinson/train.csv"
        test_path = "data/parkinson/test.csv"

        tr = pd.read_csv(train_path)
        te = pd.read_csv(test_path)

        Xtr = tr.iloc[:, :-1].values
        ytr = tr.iloc[:, -1].values

        Xte = te.iloc[:, :-1].values
        yte = te.iloc[:, -1].values

        scaler = MinMaxScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        features = tr.columns[:-1].to_numpy()

        return Xtr, Xte, ytr, yte, features


    elif name == "cancer":

        path = "data/cancer_gene_expression/cancer.csv"
        df = pd.read_csv(path)

        # =========================
        # SPLIT X / y
        # =========================
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # =========================
        # LABEL ENCODING (SAFE)
        # =========================
        if y.dtype == "object" or isinstance(y[0], str):
            y = LabelEncoder().fit_transform(y)

        y = np.array(y).ravel()

        # =========================
        # SCALE FEATURES
        # =========================
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # =========================
        # TRAIN / TEST SPLIT
        # =========================
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=SEED
        )

        features = df.columns[:-1].to_numpy()

        return Xtr, Xte, ytr, yte, features


    else:
        raise ValueError(
            f"Unsupported dataset: {name}. Use 'parkinson' or 'cancer'."
        )