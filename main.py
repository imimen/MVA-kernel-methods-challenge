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

baselines = {
    "ridge": kernelRidge,
    "svm": kernelSVM,
}


def main(args):
    clf = baselines[args.baseline.lower()](
        kernel_type=args.kernel.lower(), d=args.d, sigma=args.sigma, c=args.c
    )

    index = []
    pred = []
    for i in range(N):
        feature_type = args.features.lower()
        if feature_type == "spectrum":
            xtrain, ytrain, xtest, ids = load_file(i)
            xtrain = seqToSpec(xtrain, args.k)
            xte = seqToSpec(xtest, args.k)
        
        if feature_type == "bow":
            xtrain, ytrain, xte, ids = load_bow(i)
        if feature_type == "fusion":
            xtrain1, ytrain, xte1, ids = load_bow(i)
            xtrain2, ytrain2, xtest2, ids2 = load_file(i)
            xtrain2 = seqToSpec(xtrain2, args.k)
            xte2 = seqToSpec(xtest2, args.k)
            xtrain = np.concatenate((xtrain1,xtrain2),axis=1)
            xte = np.concatenate((xte1,xte2),axis=1)
        """
        size = int(0.7 * xtrain.shape[0])
        xtr, xval, ytr, yval = (
            xtrain[:size],
            xtrain[size:],
            ytrain[:size],
            ytrain[size:],
        )
        """
        scaler = StandardScaler()
        scaler.fit(xtrain)
        xtrain = scaler.transform(xtrain)
        xte = scaler.transform(xte)
        xtr, xval, ytr, yval = train_test_split(
                    xtrain, ytrain, test_size=0.7, random_state=42)

        _ = clf.fit(xtr, ytr)
        predval = clf.predict(xtr, xval)
        print(confusion_matrix(yval, predval))
        print("accuracy is "+str(accuracy_score(yval, predval)))
        ytest = clf.predict(xtr, xte)

        index.extend(ids)
        pred.extend(ytest)

    name = args.features.lower() + "_k_" + str(args.k) + clf.type
    save_file(index, pred, name=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        choices=["spectrum", "mismatch", "substring","bow","fusion"],
        default="spectrum",
    )
    parser.add_argument(
        "--baseline", type=str, choices=["ridge", "logistic", "svm"], default="ridge"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["linear", "polynomial", "gaussian"],
        default="linear",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--d", type=int, default=2)
    args = parser.parse_args()
    main(args)