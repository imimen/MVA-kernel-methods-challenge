import argparse
import os
import numpy as np
import pandas as pd
from utils import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# global variables
DATA_FOLDER = "machine-learning-with-kernel-methods-2021"
TRAIN_FILE = "Xtr{}.csv"
LABEL_FILE = "Ytr{}.csv"
TEST_FILE = "Xte{}.csv"
N = 3  # number of datasets

HIST_FILE = "results-history.txt"

baselines = {
    "ridge": kernelRidge,
    "svm": kernelSVM,
    "logistic": KernelLogistic,
}


def main(args):

    feature_type = args.features.lower()
    fmaker = feature_extractor(
        feature_type=feature_type,
        k=args.k,
        m=args.m,
        lmbda=args.lmbda,
        order_of_fourier_kmers=args.order_of_fourier_kmers,
        nb_of_fourier_coeffs=args.order_of_fourier_kmers,
    )

    kernel_type = args.kernel.lower() # if feature_type == "bow" else "linear"
    
    clf = Baseline(
        baseline_type=args.baseline.lower(), 
        kernel_type=kernel_type, 
        d=args.d, 
        sigma=args.sigma, 
        c=args.c)
    
    index = []
    pred = []
    accs = []
    print(f"Parameters of the model : \nfeatures : {args.features.lower()} \nbaseline+kernel : {clf.type}")
    for i in range(N):
        xtrain, ytrain, xte, ids = fmaker(i)

        scaler = StandardScaler()
        scaler.fit(xtrain)
        xtrain = scaler.transform(xtrain)
        xte = scaler.transform(xte)
        xtr, xval, ytr, yval = train_test_split(
            xtrain, ytrain, test_size=0.3, random_state=42
        )

        _ = clf.fit(xtr, ytr)
        predval = clf.predict(xtr, xval)
        print(confusion_matrix(yval, predval))
        
        acc = accuracy_score(yval, predval)
        print("accuracy is " + str(acc))
        ytest = clf.predict(xtr, xte)

        index.extend(ids)
        pred.extend(ytest)
        accs.append(acc)
        
    name = args.features.lower() + "_k_" + str(args.k) + clf.type
    save_file(index, pred, name=name)
    save_results(name, accs, sum(accs)/N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        choices=["spectrum", "mismatch", "substring", "bow", "fourier", "fusion"],
        default="spectrum",
    )
    parser.add_argument(
        "--baseline", type=str, choices=["ridge", "logistic", "svm"], default="svm"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["linear", "polynomial", "gaussian"],
        default="linear",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--lmbda", type=float, default=2.0)
    parser.add_argument("--c", type=float, default=0.0001)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--nb_of_fourier_coeffs", type=int, default=5)
    parser.add_argument("--order_of_fourier_kmers", type=int, default=1)
    args = parser.parse_args()
    main(args)