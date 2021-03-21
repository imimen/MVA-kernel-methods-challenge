import numpy as np
import pandas as pd
import json
import os
from itertools import product, combinations
from scipy.sparse import csr_matrix
from tqdm import tqdm
from numpy.fft import rfft, rfftfreq


# global variables
DATA_FOLDER = "machine-learning-with-kernel-methods-2021"
RES_FOLDER = "results"
TRAIN_FILE = "Xtr{}.csv"
LABEL_FILE = "Ytr{}.csv"
TEST_FILE = "Xte{}.csv"
PRED_FILE = "submission_{}.csv"
HIST_FILE = "results-history.txt"
N = 3  # number of datasets


# UTILS
def load_seq(index):
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
    df_train = pd.read_csv(
        os.path.join(DATA_FOLDER, TRAIN_FILE.format(str(index) + "_mat100")),
        header=None,
        sep="\s+",
    )
    df_label = pd.read_csv(os.path.join(DATA_FOLDER, LABEL_FILE.format(str(index))))
    df_test = pd.read_csv(
        os.path.join(DATA_FOLDER, TEST_FILE.format(str(index) + "_mat100")),
        header=None,
        sep="\s+",
    )
    df_test_id = pd.read_csv(os.path.join(DATA_FOLDER, TEST_FILE.format(index)))
    xtrain = df_train.to_numpy()
    ytrain = df_label["Bound"].values
    xtest = df_test.to_numpy()
    ids = df_test_id["Id"].values
    return xtrain, ytrain, xtest, ids


def save_file(ids, pred, name):
    global RES_FOLDER, PRED_FILE
    df_pred = pd.DataFrame({"Id": ids, "Bound": pred})
    df_pred.to_csv(os.path.join(RES_FOLDER, PRED_FILE.format(name)))
    
    
def save_results(model, results, mean):
    global HIST_FILE
    line = '\n' + model + '   ' + str(results) + '    mean: ' + str(mean) 
    with open(HIST_FILE,"a+") as f:
        f.write(line)


def getVocab(k):
    chars = ["A", "C", "G", "T"]
    chars = "".join(set(chars))
    vocab = ["".join(c) for c in product(chars, repeat=k)]
    return vocab


def getKmers(x, k=3):
    return [x[i : i + k].upper() for i in range(len(x) - k + 1)]


def m_match(kmer, candidate, m=1):
    """
    Return True if it is a match up to m mistakes, else False
    """
    if len(kmer) != len(candidate):
        raise ValueError("problem in length of kmer and candidate")
    errors = 0
    for unit1, unit2 in zip(kmer, candidate):
        if unit1 != unit2:
            errors += 1
    return errors <= m


def make_PCA(features):
    """
    Do the PCA of a set a vectors features
    features is of shape (n_vectors, dim_vectors)
    """
    return


# FEATURE EXTRACTORS
class to_spectrum:
    def __init__(self, k=3):
        self.type = f"_{k}-spectrum"
        self.k = k
        self.vocab = getVocab(k)

    def getFeatures(self, x):
        features = np.zeros((x.shape[0], len(self.vocab)), dtype=np.int8)
        for idx, seq in enumerate(x):
            kmers = getKmers(seq, self.k)
            kmers, counts = np.unique(kmers, return_counts=True)
            for i in range(len(kmers)):
                j = self.vocab.index(kmers[i])
                features[idx, j] = counts[i]
        return features

    def __call__(self, xtrain, xtest):
        xtrain = self.getFeatures(xtrain)
        xte = self.getFeatures(xtest)
        return xtrain, xte


class to_mismatch:
    def __init__(self, k=3, m=1):
        self.type = f"_{k}_{m}_mismatch"
        self.k = k
        self.m = m
        self.vocab = getVocab(k)

    def getFeatures(self, x):
        features = np.zeros((x.shape[0], len(self.vocab)), dtype=np.int8)
        for idx, seq in enumerate(x):
            kmers = getKmers(seq, self.k)
            for kmer in kmers:
                # Let's find each unit of vocab corresponding up to m mismatches
                for ind_cand, candidate in enumerate(self.vocab):
                    if m_match(kmer, candidate, self.m):
                        features[idx, ind_cand] += 1
        return features

    def __call__(self, xtrain, xtest):
        xtrain = self.getFeatures(xtrain)
        xte = self.getFeatures(xtest)
        return xtrain, xte


class to_fourier:
    def __init__(self, order_of_kmers=3, nb_of_coeffs=5):
        """
        If order_of_kmers = 1, returns the fourier transform of the vectors of indicator of
        1-mers (vector indicator of A, vector indicator of T ...etc)
        If order_of_kmers = 2, returns the fourier transform of the vectors of indicator of
        1-mers (vector indicator of AA, vector indicator of AT, of AC, AG, TA,..etc...)
        nb_of_coeffs sets the number of maximum coefficients we consider in the fourier transform
        """
        self.type = f"_{order_of_kmers}_{nb_of_coeffs}_fourier"
        self.order_of_kmers = order_of_kmers
        self.nb_of_coeffs = nb_of_coeffs
        self.vocab = getVocab(order_of_kmers)

    def get_indicator_of_pattern_sequence(self, seq_of_tokens, pattern):
        """
        takes as input the sequence of tokens (an array of strings)
        if len(pattern) = 2, the seq_of_tokens = [AT,AG,CG...etc...]
        """
        mask = seq_of_tokens == pattern
        return mask * 1

    def getFeatures(self, x):
        features = list()
        for seq in tqdm(x):
            seq_of_tokens = np.array(getKmers(seq, k=self.order_of_kmers))
            seq_feature = []
            for pattern in self.vocab:
                indi = self.get_indicator_of_pattern_sequence(seq_of_tokens, pattern)
                fourier = abs(rfft(indi)) ** 2
                idx = (-fourier).argsort()[: self.nb_of_coeffs]
                fourier[::-1].sort()
                amplitudes = fourier[:5]
                seq_feature += list(idx) + list(amplitudes)
            features.append(seq_feature)
        return np.array(features)

    def __call__(self, xtrain, xtest):
        xtrain = self.getFeatures(xtrain)
        xte = self.getFeatures(xtest)
        return xtrain, xte


class to_substring:
    def __init__(self, k=3, lmbda=2):
        self.type = f"_substring_{k}_{lmbda}"
        self.k = k
        self.lmbda = lmbda
        self.vocab = self.getVocab(k)
        self.indices = None
        self.lengths = None
        self.n = None

    def getKindices(self):
        indices = [list(c) for c in combinations(np.arange(self.n), self.k)]
        lengths = [idx[-1] - idx[0] + 1 for idx in indices]
        # print(lengths)
        return indices, lengths

    def getFeatures(self, x):
        n = len(x[0])  # length of an ADN sequence
        if self.indices is None or n != self.n:
            self.n = n
            self.indices, self.lengths = self.getKindices()
        features = np.zeros((x.shape[0], len(self.vocab)), dtype=np.int8)
        for idx, seq in enumerate(x):
            words = [seq[i[0] : i[-1]] for i in self.indices]
            for j, word in enumerate(self.vocab):
                # print(words, word)
                features[idx, j] = self.lmbda ** sum(self.lengths[words == word])
        return features

    def __call__(self, xtrain, xtest):
        xtrain = self.getFeatures(xtrain)
        xte = self.getFeatures(xtest)
        return xtrain, xte
    
    
# FEATURE EXTRACTION STRATEGIES

load_file = {"bow": load_bow, "seq": load_seq, "both": (load_bow, load_seq)}


class feature_extractor:
    def __init__(
        self,
        feature_type,
        k=3,
        m=1,
        lmbda=2,
        order_of_fourier_kmers=1,
        nb_of_fourier_coeffs=5,
    ):

        # By defaut, load original sequences, then change this method if feature_type="bow" or "fusion"
        self.load_file = load_file["seq"]
        self.is_fusion = False

        if feature_type == "spectrum":
            self.to_vec = to_spectrum(k)
        elif feature_type == "mismatch":
            self.to_vec = to_mismatch(k, m)
        elif feature_type == "substring":
            self.to_vec = to_substring(k, lmbda)
        elif feature_type == "fourier":
            self.to_vec = to_fourier(order_of_fourier_kmers, nb_of_fourier_coeffs)
        elif feature_type == "bow":
            self.to_vec = None
            self.load_file= load_file["bow"]
        elif feature_type == "fusion":
            self.is_fusion = True
            self.to_vec = (to_spectrum(k), to_mismatch(k, m), to_fourier(order_of_fourier_kmers, nb_of_fourier_coeffs))     
        
        if self.to_vec is None:
            name = ''
        elif self.is_fusion == True:
            name = "_"+ str([self.to_vec[i].type for i in range(len(self.to_vec))])
        else:
            name = self.to_vec.type
        self.type = f"_{name}"

    def __call__(self, i):
        if self.is_fusion == True:
            xtrain1, ytrain, xtest1, ids = load_bow(i)
            
            xtrain_ref, ytrain_ref, xtest_ref, ids_ref = load_seq(i)
            
            xtrain2, xtest2 = self.to_vec[0](xtrain_ref, xtest_ref)
            xtrain3, xtest3 = self.to_vec[1](xtrain_ref, xtest_ref)
            xtrain4, xtest4 = self.to_vec[2](xtrain_ref, xtest_ref)
            
            xtrain = np.concatenate((xtrain1, xtrain2, xtrain3, xtrain4), axis=1)
            xtest = np.concatenate((xtest1, xtest2, xtest3, xtest4), axis=1)
        else :
            xtrain, ytrain, xtest, ids = self.load_file(i)
            if self.to_vec is not None:
                xtrain, xtest = self.to_vec(xtrain, xtest)
        return xtrain, ytrain, xtest, ids
    
    
    