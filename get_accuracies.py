#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

import numpy as np
import pandas
import glob

# for each folder (mnist, cifar, etc...) find systematic way of pulling out
# result of last layer. compare this with training labels from .csv input.

# get result of last layer from (DATA)
# get training labels from .CSV (DATA) folder
# for each architecture:
# accuracy w/ eigen
# accuracy w/ gemmlowp
#

def main():
    #############
    # FUNCTIONS #
    #############
    def build_results(path):
        dirs = sorted(glob.glob(path))
        
        results = []
        for f in dirs:
            result = pandas.read_csv(f, header=None)    
            result = np.argmax(result.values[0])
            results.append(result)
        
        results = np.array(results)

        return results
                    
    def score(a, b):
        z = 1
        c = a - b
        for i in c:
            if i != 0:
                z += 1
    
        z = float(z) / len(a)
        return z

    #########
    # MNIST #
    #########
    # Actual values
    mnist_train = pandas.read_csv("data/mnist/mnist_train_100.csv", header=None)
    mnist_labels = mnist_train.values[:,0]

    # Eigen Results
    mnist_eigen_results = build_results("data/mnist/features_eigen/fc2*")

    # Gemmlowp Results
    mnist_gemmlowp_results = build_results("data/mnist/features_gemmlowp/fc2*") 
   
    print score(mnist_eigen_results, mnist_labels)
    print score(mnist_gemmlowp_results, mnist_labels)

    #########
    # CIFAR #
    #########
    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    # Actual values
    cifar_train = unpickle("data/cifar10/cifar-10-batches-py/data_batch_1")
    cifar_labels = cifar_train['labels'] 

    # Eigen Results
    cifar_eigen_results = build_results("data/cifar10/features_eigen/fc2*")

    # Gemmlowp Results
    cifar_gemmlowp_results = build_results("data/cifar10/features_gemmlowp/fc2*") 

    print score(cifar_eigen_results, mnist_labels)
    print score(cifar_gemmlowp_results, mnist_labels)

if __name__ == "__main__":
    main()
