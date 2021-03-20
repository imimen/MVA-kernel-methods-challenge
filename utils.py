import numpy as np
import pandas as pd
import os
from itertools import product
from scipy.sparse import csr_matrix
from tqdm import tqdm
from numpy.fft import rfft, rfftfreq


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


def load_bow(index):
    global DATA_FOLDER, TRAIN_FILE, TEST_FILE
    df_train = pd.read_csv(os.path.join(DATA_FOLDER, TRAIN_FILE.format(str(index) + "_mat100")),
                           header=None,sep='\s+')
    df_label = pd.read_csv(os.path.join(DATA_FOLDER, LABEL_FILE.format(str(index))))
    df_test = pd.read_csv(os.path.join(DATA_FOLDER, TEST_FILE.format(str(index) + "_mat100")),
                          header=None,sep='\s+')
    df_test_id = pd.read_csv(os.path.join(DATA_FOLDER, TEST_FILE.format(index)))
    xtrain = df_train.to_numpy()
    ytrain = df_label["Bound"].values
    xtest = df_test.to_numpy()
    ids = df_test_id["Id"].values
    return xtrain, ytrain, xtest, ids


def save_file(ids, pred, name):
    global DATA_FOLDER, PRED_FILE
    df_pred = pd.DataFrame({"Id": ids, "Bound": pred})
    df_pred.to_csv(os.path.join(DATA_FOLDER, PRED_FILE.format(name)))


def getVocab(x, k):
    # chars = []
    # for seq in x:
    #    chars.extend(list(set(seq)))
    chars = ["A", "C", "G", "T"]
    chars = "".join(set(chars))
    vocab = ["".join(c) for c in product(chars, repeat=k)]
    return vocab


def getKmers(x, k=3):
    return [x[i : i + k].upper() for i in range(len(x) - k + 1)]


def seqToSpec(x, k=3):
    vocab = getVocab(x, k)
    vocab.sort()
    features = np.zeros((x.shape[0], len(vocab)), dtype=np.int8)
    for idx, seq in tqdm(enumerate(x)):
        kmers = getKmers(seq, k)
        kmers, counts = np.unique(kmers, return_counts=True)
        for i in range(len(kmers)):
            j = vocab.index(kmers[i])
            features[idx, j] = counts[i]
    return features


def m_match(kmer,candidate,m):
    """
    Return True if it is a match up to m mistakes, else False
    """
    if len(kmer) != len(candidate):
        raise ValueError("problem in length of kmer and candidate")
    errors = 0
    for unit1, unit2 in zip(kmer,candidate):
        if unit1 != unit2:
            errors += 1
    if errors > m:
        return False
    else:
        return True


def mismatch(x, k=3, m = 1):
    vocab = getVocab(x, k)
    vocab.sort()
    features = np.zeros((x.shape[0], len(vocab)), dtype=np.int8)
    for idx, seq in tqdm(enumerate(x)):
        kmers = getKmers(seq, k)
        for kmer in kmers:
            #Let's find each unit of vocab corresponding up to m mismatches
            for ind_cand, candidate in enumerate(vocab):
                if m_match(kmer,candidate,m):
                    features[idx, ind_cand] += 1
    return features


def make_PCA(features):
    """
    Do the PCA of a set a vectors features
    features is of shape (n_vectors, dim_vectors)
    """
    return


def get_tf(x,order_of_kmers = 2, nb_of_coeffs = 5):
    """
    If order_of_kmers = 1, returns the fourier transform of the vectors of indicator of
    1-mers (vector indicator of A, vector indicator of T ...etc)
    If order_of_kmers = 2, returns the fourier transform of the vectors of indicator of
    1-mers (vector indicator of AA, vector indicator of AT, of AC, AG, TA,..etc...)
    nb_of_coeffs sets the number of maximum coefficients we consider in the fourier transform
    """
    vocab = getVocab(x, order_of_kmers)
    features = list()
    for seq in tqdm(x):
        seq_of_tokens = np.array(getKmers(seq,k=order_of_kmers))
        seq_feature = []
        for pattern in vocab:
            indi = get_indicator_of_pattern_sequence(seq_of_tokens,pattern)
            fourier = abs(rfft(indi)) ** 2
            idx = (-fourier).argsort()[:nb_of_coeffs]
            fourier[::-1].sort()
            amplitudes = fourier[:5]
            seq_feature += list(idx) + list(amplitudes)
        features.append(seq_feature)
    return np.array(features)
        

def get_indicator_of_pattern_sequence(seq_of_tokens,pattern):
    """
    takes as input the sequence of tokens (an array of strings)
    if len(pattern) = 2, the seq_of_tokens = [AT,AG,CG...etc...]
    """
    mask = seq_of_tokens == pattern
    return mask*1