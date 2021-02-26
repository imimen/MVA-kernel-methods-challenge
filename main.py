import argparse
import os 
import numpy as np
import pandas as pd
from utils import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# global variables
DATA_FOLDER = "machine-learning-with-kernel-methods-2021"
TRAIN_FILE = "Xtr{}.csv"
LABEL_FILE = "Ytr{}.csv"
TEST_FILE = "Xte{}.csv"
N = 3 # number of datasets
    

def main(args):
    # model parameters
    features = args.features.lower()
    kernel = args.kernel.lower()
    baseline = args.baseline.lower()
    k = int(args.k)

    if baseline == "ridge":
        clf = kernelRidge(kernel)
    
    name = baseline+"_"+kernel+"_"+str(k)+"_"+features

    index = []
    pred = []
    for i in range(N):
        xtrain, ytrain, xtest, ids = load_file(i)
        xtrain, xte = seqToSpec(xtrain, k), seqToSpec(xtest, k)
        size = int(0.7 * xtrain.shape[0])
        xtr, xval, ytr, yval = xtrain[:size], xtrain[size:], ytrain[:size], ytrain[size:]
        _ = clf.fit(xtr, ytr)
        predval = clf.predict(xtr, xval)
        predval = threshold(predval)
        print(confusion_matrix(yval, predval))
        
        ytest = clf.predict(xtr, xte)
        ytest = threshold(ytest)
        index.extend(ids)
        pred.extend(ytest)
    save_file(index, pred, name=name)
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",  choices=["spectrum", "mismatch", "substring"], default="spectrum")
    parser.add_argument("--baseline",  choices=["ridge", "logistic", "svm"], default="ridge")
    parser.add_argument("--kernel",  choices=["linear", "polynomial", "gaussian", "rbf"], default="linear")
    parser.add_argument("--k",  choices=['3', '4', '5', '6'], default='3')
    args = parser.parse_args()
    main(args)
