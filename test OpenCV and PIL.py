# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, glob, os
import numpy as np

import matplotlib as mpl
mpl.use('module://IPython.zmq.pylab.backend_inline')
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['image.cmap'] = 'jet'

sys.path.append("/extra/InVivoDog/python/cine/tools")

from dogdata import DogData
from cine import Cine

import xlrd

# <codecell>

import cv2

# <codecell>

cv2.__version__

# <codecell>

cv2.__dict__

# <codecell>

cv2.cv.__dict__

# <codecell>

cv2.cv.__version__

# <codecell>

help cv2.findCirclesGrid

# <codecell>

help cv2.findCirclesGridDefault

# <codecell>

cv2.CALIB_CB_ASYMMETRIC_GRID

# <codecell>

help cv2.FeatureDetector_create

# <codecell>

help cv2.goodFeaturesToTrack

# <codecell>

print cv2.getBuildInformation()

# <codecell>

# old module
import cv2.cv

# <codecell>

im = cv2.imread('/extra/InVivoDog/python/cine/Computer Vision with Python/data/empire.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# <codecell>

print im.dtype
print gray.dtype
print hsv.dtype

# <codecell>

print im.shape
print gray.shape
print hsv.shape

# <codecell>

# mpl.rcParams['image.cmap'] = 'jet'

fig, ax = plt.subplots(figsize = (14, 6))

# for RGB values cmap is being ignored
ax.imshow(im, origin = 'upper', cmap = mpl.cm.jet)

fig, ax = plt.subplots(figsize = (14, 6))

ax.imshow(im[:, :, 0], origin = 'upper', cmap = mpl.cm.jet)

fig, ax = plt.subplots(figsize = (14, 6))

ax.imshow(gray, origin = 'upper', cmap = mpl.cm.jet)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (14, 6))

ax.imshow(hsv[:, :, 2], origin = 'upper', cmap = mpl.cm.gray)

plt.show()

# <codecell>

# 16 bit to 8 bit conversion

def conv16to8(frame):
    """
    convert pixel values from 16 bit to 8 bit and rescale to use the entire range of gray values
    input: frame as 16 bit image (or arbitrary accuracy)
    output: 8 bit image array (uint8)
    """
    fmin = np.double(frame.min())
    fmax = np.double(frame.max())
    
    return np.uint8( (2.**8 - 1.0) / (fmax - fmin) * (frame - fmin) )

# <codecell>

from PIL import Image

# <codecell>

help Image.open

# <codecell>

help Image.Image.convert

# <codecell>

pil_im = Image.open('/extra/InVivoDog/python/cine/Computer Vision with Python/data/empire.jpg')

# <codecell>

plt.imshow(np.array(pil_im), origin = 'upper')
plt.show()

# <codecell>

plt.imshow(np.array(pil_im.convert('L')), origin = 'upper', cmap = mpl.cm.gray)
plt.show()

# <codecell>

def histeq(image, number_bins = 256):
    """ Histogram equalization of a grayscale image. 
        Input:   image as numpy array

        Return:  histogram-equd image, cdf """

    imhist, bins = np.histogram(image.flatten(), number_bins, normed = True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = (2**8 - 1) * cdf / cdf[-1] # normalize
    
    # linear interpolation of cdf to find nearest pixel values
    im2 = np.interp(image.flatten(), bins[:-1], cdf)
    return im2.reshape(image.shape), cdf

# <codecell>

def pca(X, centerdata = True):
    """ Principal Component Analysis
        Input:  X, matrix with training data stored as flattened arrays in rows
        Return: projection matrix (with important dimensions first), variance, mean """
    
    # get dimensions
    num_data, dim = X.shape
    num_images = num_data
    # rows: num_data: different images, data vectors
    num_pixels = dim
    # columns: dim: number of pixels (observations) for each observation (image) data
    
    mean_X = X.mean(axis = 0)
    if centerdata:
        # center data
        X = X - mean_X
    
    if dim > num_data:
        # PCA - compact trick
        # calculate chronos first, shooting method
        # chronos: with respect to time, i.e. as in a movie: different images in the image list
        
        # covariance matrix: averaging over (along) pixels
        # result is a small matrix of the size of the number of images (num_data)
        M = np.dot( X, X.T )
        
        # eigenvalues and eigenvectors of Hermitian (symmetric) matrix
        eigval, eigvec = np.linalg.eigh(M) # eigvals NOT ordered
        
        # compact trick: project the data back onto the chronos to get the topos
        tmp = np.dot( X.T, eigvec ).T
        
        # reverse: eigenvalues are in increasing order???
        S = sqrt(eigval)[::-1]
        # reverse: last eigenvector is most important???
        topos = tmp[::-1]
        
        # somehow normalize topos
        for i in range(topos.shape[1]):
            topos[:,i] /= S
        V = topos
    else:
        # use regular PCA with SVD
        U, S, V = np.linalg.svd(X)
        
        # only return the first num_data, i.e. as the number of images
        V = V[:num_data]
    
    # return the projection matrix, the variance, the mean
    return V, S, mean_X

# <codecell>


