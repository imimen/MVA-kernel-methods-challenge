# Kernel Methods for Machine Learning inclass data challenge, 2020-2021 

This repository contains the handing for the data challenge as part of the MVA's course "machine learning with kernel methods".
The goals is to implement machine learning algorithms and adapt them to structural data, without using any external ML libraries (libsvm, liblinear, scikit-learn,..)
Only general purpose libraries such as the ones for linear algebra or optimization are allowed.

### Challenge :
The challenge deals with a sequence classification task: predicting whether a DNA sequence region is binding site to a specific transcription factor TF. Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes. Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.

This challenge proposes three datasets, corresponding to three different TFs and thus should be predicted seperately. The different datasets are provided in two formats:

- the raw sequences of DNA : with ***Xtr{i}.csv*** being the training sequences and  ***Xte{i}.csv*** the test ones. ***i=0,1,2***
- A bag of words based-representation of the previous sequences : in the files ***Xtr{i}_mat100.csv*** (train) and ***Xte{i}_mat100.csv*** (test) ***i=0,1,2***

For further details, here is the link for the data challenge : [Kaggle challenge](https://www.kaggle.com/c/machine-learning-with-kernel-methods-2021/)  

### Implemented baselines:

In this repository, we included in ***models.py*** our implementation for the basic kernels: **linear, polynomial and gaussian** along with some classification baselines: **kernel linear regression**, **logistic regression** and **SVM**. Provided that we are dealing with a classification task related to **DNA sequences**, we explored different strategies for extracting features from this particular data structure. Precisely, we worked with **spectral**, **mismatch** and **fourier features**, besides the provided Bag of word based representations. All the utils related to the preprocessing of the data figures in ***utils.py***.

In order to test the different baselines we have implemented so far, you need to run the following script:

```
python main.py --args $ARGS
```
Where ARGS refer to the set of appropriate parameters (depending on the chosen features, kernel and classification baseline) including :

- features: "spectrum", "mismatch", "bow", "fourier", "fusion" (the possible choices)
- baselines : "ridge", "logistic", "svm" (the possible choices)
- kernels : "linear", "polynomial", "gaussian" (the possible choices)
- k : length of k-mers (fixed length subsequences)
- m : number of allowed mismatches
- nb_of_fourier_coeffs : number of coefficients to use for the fourier features
- order_of_fourier_kmers :  length of kmers used for the fourier features
- c : regularization parameter for the baseline
- sigma : variance of the gaussian kernel
- d : degree of polynomial kernel 
    
Please make sure to put the different scripts ***models.py***, ***utils.py*** and ***main.py*** as well as the the dataset: ***/machine-learning-with-kernel-methods-2021*** in the same folder.


