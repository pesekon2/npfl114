#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    ddist = list()
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            ddist.append(line)

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.
    ddist_unique = list(set(ddist))
    npddist = np.zeros(len(ddist_unique))
    for i in range(len(ddist_unique)):
        npddist[i] += ddist.count(ddist_unique[i]) / (len(ddist))

    # Load model distribution, each line `word \t probability`.
    mdist = dict()
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            line = line.split('\t')
            mdist.update({line[0]: line[1]})

    # TODO: Create a NumPy array containing the model distribution.
    npmdist = np.zeros(len(ddist_unique))
    mdist_keys = mdist.keys()
    for i in ddist_unique:
        if i in mdist_keys:
            npmdist[ddist_unique.index(i)] += float(mdist[i])

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - np.sum(npddist * np.log(npddist))
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    np.seterr(divide='ignore')
    H_PQ = - np.sum(npddist * np.log(npmdist))
    print("{:.2f}".format(H_PQ))
    D_PQ = H_PQ - entropy
    print("{:.2f}".format(D_PQ))
