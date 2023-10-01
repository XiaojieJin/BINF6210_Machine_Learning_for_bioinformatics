import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PCA:

    def __init__(self,mat):
        if type(mat) is np.ndarray:
            self._mat = mat
            self.result = self.Result()
        else:
            raise ValueError
    
    class Result:
        def __init__(self):
            self.score = np.zeros((2,3))
            self.loadings = np.zeros((2,3))
            self.percant_variance_explained = np.zeros((2,3))
            self.variance_explained = np.zeros((2,3))
       
    def fit_transform(self):
        #step1: Do mean-centering
        mean_array = np.mean(self._mat, axis=0)
        mat_mean =self._mat - mean_array

        #step2: Compute the convariance matrix
        #(we can't use np.cov() directly because np.cov includes mean-centering)
        #mat_cov = np.cov(self._mat.T)
        mat_cov = np.matmul(mat_mean.T,mat_mean)/(np.shape(self._mat)[0] - 1)
        #return mat_cov

        #step3: Perform egen-decomposition
        #we should use eigh rather than eig to avoid complex number
        #S: eigenvalue; U: eigenvector 
        S, U = np.linalg.eigh(mat_cov)
        self.result.variance_explained = sorted(S,reverse=True)
        self.result.percant_variance_explained = sorted(S,reverse=True)/np.sum(S)
        self.result.loadings = U
    
        #step4: Project the data onto the principal component axes
        self.result.score = np.matmul(self._mat, U)
        return self.result
    
    



