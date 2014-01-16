# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os, csv, glob
import numpy as np

# %pylab inline

import matplotlib as mpl
# use the backend for inline plots in the IPython Notebook
# so then don't need to use magic pylab with inline option
mpl.use('module://IPython.zmq.pylab.backend_inline')

import matplotlib.pyplot as plt

print "Matplotlib will use backend: ", mpl.get_backend()
print "Pyplot will use backend: ", plt.get_backend()

# <codecell>

mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['image.cmap'] = 'jet'

# <codecell>

import pandas as pd
pd.__version__

# <codecell>

d = pd.read_csv("/extra/InVivoDog/Dinesh/ruhlen_short_distancematrix.csv", header = None)

# <codecell>

from IPython.display import HTML
HTML(d.to_html())

# <codecell>

dmat = np.array(d)

# <codecell>

plt.matshow(dmat)

plt.colorbar()

plt.show()

# <codecell>

plt.matshow(dmat - dmat.mean(axis = 0))

plt.colorbar()

plt.show()

# <codecell>

figsize_current = mpl.rcParams['figure.figsize']
figsize_orig = mpl.rcParamsOrig['figure.figsize']
figsize_default = mpl.rcParamsDefault['figure.figsize']

print "current: ", figsize_current
print "original: ", figsize_orig
print "default: ", figsize_default

# <codecell>

m = np.array([[1,2], [3,4]])

print m

print m.mean(axis = 0)

# <codecell>

dmat_sorted = np.sort(dmat, axis = 0)

s = np.argsort(dmat_sorted.mean(axis = 0))

fig, ax = plt.subplots(figsize = (10, 8))

axmat = ax.matshow(dmat_sorted[:, s])

fig.colorbar(axmat)

plt.show()

# <codecell>

import sklearn

# <codecell>

# sklearn.decomposition.KernelPCA

# <codecell>

import numpy as np
import numpy.linalg as npl

def PCA(datamatrix, centering = False):
    """
    Performs a PCA, a Principal Component Analysis
    
    datamatrix: different variables along columns, different observations along rows
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
    

# <codecell>

centering = True

eigval, norm_eigval, eigvec, projection = PCA(dmat, centering = centering)

# <codecell>

plt.figure(figsize = (14, 2))

plt.plot(dmat.mean(axis = 0), 'g.-')

plt.title('mean of data in each column along rows')

plt.show()

# <codecell>

plt.figure(figsize = (14, 20))

for k in range(10):
    plt.subplot(10, 1, k + 1)
    
    plt.plot(eigvec[:, k], 'r.-')

plt.show()

# <codecell>

Nrows, Ncols = dmat.shape

Ncomp = 1

reconstruction = np.dot(projection[:, 0:Ncomp], eigvec[:, 0:Ncomp].T)

fullreconstruction = np.dot(projection, eigvec.T)

if centering:
    reconstruction += np.tile( dmat.mean(axis = 0), (Nrows, 1) )
    fullreconstruction += np.tile( dmat.mean(axis = 0), (Nrows, 1) )

plt.figure()
plt.matshow(dmat)
plt.title('original data')
plt.colorbar()

plt.figure()
plt.matshow(reconstruction)
plt.title('filtered data: Ncomp = %d' % Ncomp)
plt.colorbar()

residuals = np.abs(reconstruction - dmat)

plt.figure()
plt.matshow(residuals)
plt.title('residuals')
plt.colorbar()

plt.show()

# <codecell>

print "mean: ", residuals.mean()
print "std: ", residuals.std()
print "max: ", residuals.max()

# <codecell>

plt.figure(figsize = (14, 20))

for k in range(10):
    plt.subplot(10, 1, k + 1)
    
    plt.plot(projection[:, k], 'b.-')

plt.show()

# <codecell>

plt.figure(figsize = (14, 10))

plt.plot(np.log10(norm_eigval * 100), 'r.-')

plt.xlim(xmin = -5)
plt.ylim(ymax = 2.5)

plt.xlabel('component index')
plt.ylabel('log10(normalized variance [%])')

plt.show()

# <codecell>

plt.figure(figsize = (14, 10))

plt.plot(np.cumsum(norm_eigval) * 100, 'r.-')

plt.xlim(xmax = 100)

plt.xlabel('component index')
plt.ylabel('cumulative sum: explained variance [%]')

plt.show()

# <codecell>


