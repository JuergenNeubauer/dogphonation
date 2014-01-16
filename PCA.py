# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import numpy.linalg as npl

def PCA(datamatrix, centering = False):
    """
    Performs a PCA, a Principal Component Analysis
    
    datamatrix: different variables in/along columns, different observations in/along rows
    centering: remove the mean of each column, so calculate the mean along/down each row for each column
    
    returns: eigval, norm_eigval, eigvec, projection_vec
    """
    
    Nrows, Ncols = datamatrix.shape
    
    # calculate mean along/down each row for each column
    mean_dat = np.mean(datamatrix, axis = 0)

    mean_datamatrix = np.tile( mean_dat, (Nrows, 1) )
    
    if centering:
        centered = datamatrix - mean_datamatrix
    else:
        centered = datamatrix
        
    # correlation matrix, summation over/along the rows in each column of the data matrix
    corr = np.dot(centered.T, centered)
    
    # weights (eigenvalues) and normalized eigenvectors of Hermitian or symmetric matrix
    # eigenvalues are not ordered in general
    # eigenvectors are in columns k, i.e. v[:, k]
    w, v = npl.eigh(corr)
    
    # sort from largest to smallest
    sortindex = np.argsort(w)[::-1]
    
    # sort eigenvectors
    eigen_vec = v[:, sortindex]

    # sort from largest to smallest
    eigen_val = sorted(w, reverse = True)
    # normalize eigenvalues
    norm_eigen_val = eigen_val / np.sum(eigen_val)
    
    # bi-orthogonal vectors from projection of data onto PCA directions, vectors are in columns
    projection_vec = np.dot(centered, eigen_vec)
    
    return eigen_val, norm_eigen_val, eigen_vec, projection_vec

