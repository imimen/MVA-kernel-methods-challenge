# Kernel Methods for Machine Learning inclass data challenge, 2020-2021 

This repository contains the handing for the data challenge as part of the MVA's course "machine learning with kernel methods".
The goals is to implement machine learning algorithms and adapt them to structural data, without using any external ML libraries (libsvm, liblinear, scikit-learn,..)
Only general purpose libraries such as the ones for linear algebra or optimization are allowed.

## Challenge :
The challenge deals with a sequence classification task: predicting whether a DNA sequence region is binding site to a specific transcription factor TF. Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes. Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.

This challenge proposes three datasets, corresponding to three different TFs and thus should be predicted seperately. The different datasets are provided in two formats:

- the raw sequences of DNA : with ***Xtr{i}.csv*** being the training sequences and  ***Xte{i}.csv*** the test ones. ***i=0,1,2***
- A bag of words based-representation of the previous sequences : in the files ***Xtr{i}_mat100.csv*** (train) and ***Xte{i}_mat100.csv*** (test) ***i=0,1,2***

For further details, here is the link for the data challenge : [Kaggle challenge](https://www.kaggle.com/c/machine-learning-with-kernel-methods-2021/)  
