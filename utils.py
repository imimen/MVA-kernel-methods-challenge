import numpy as np
import pandas as pd
import os
from itertools import product
from scipy.sparse import csr_matrix


# global variables
DATA_FOLDER = "machine-learning-with-kernel-methods-2021"
TRAIN_FILE = "Xtr{}.csv"
LABEL_FILE = "Ytr{}.csv"
TEST_FILE = "Xte{}.csv"
PRED_FILE = "submission_{}.csv"
N = 3  # number of datasets


def load_file(index):
    global DATA_FOLDER, TRAIN_FILE, TEST_FILE
    df_train = pd.read_csv(os.path.join(DATA_FOLDER, TRAIN_FILE.format(index)))
    df_label = pd.read_csv(os.path.join(DATA_FOLDER, LABEL_FILE.format(index)))
    df_test = pd.read_csv(os.path.join(DATA_FOLDER, TEST_FILE.format(index)))

    xtrain = df_train["seq"].values
    ytrain = df_label["Bound"].values
    xtest = df_test["seq"].values
    ids = df_test["Id"].values
    return xtrain, ytrain, xtest, ids


def save_file(ids, pred, name):
    global DATA_FOLDER, PRED_FILE
    df_pred = pd.DataFrame({"Id": ids, "Bound": pred})
    df_pred.to_csv(os.path.join(DATA_FOLDER, PRED_FILE.format(name)))


def getVocab(x, k):
    chars = []
    for seq in x:
        chars.extend(list(set(seq)))
    chars = "".join(set(chars))
    vocab = ["".join(c) for c in product(chars, repeat=k)]
    return vocab


def getKmers(x, k=3):
    return [x[i : i + k].upper() for i in range(len(x) - k + 1)]


def seqToSpec(x, k=3):
    vocab = getVocab(x, k)
    vocab.sort()
    features = np.zeros((x.shape[0], len(vocab)), dtype=np.int8)
    for idx, seq in enumerate(x):
        kmers = getKmers(seq, k)
        kmers, counts = np.unique(kmers, return_counts=True)
        for i in range(len(kmers)):
            j = vocab.index(kmers[i])
            features[idx, j] = counts[i]
    return features