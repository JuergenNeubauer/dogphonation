# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, glob, os
import numpy as np

# <codecell>

import cv2

# <codecell>

# to 8 bit conversion

def convto8(frame, debug = False):
    """
    convert pixel values from any bit to 8 bit and rescale to use the entire range of gray values
    input: frame as any bit image (or arbitrary accuracy)
    output: 8 bit image array (uint8)
    """
    fmin = np.double(np.nanmin(frame))
    fmax = np.double(np.nanmax(frame))
    
    scale = 1.0
    shift = 0.0

    scale = (2.0**8 - 1.0) / (fmax - fmin)
    shift = - scale * fmin
    
    if debug:
        print "convto8: fmin: ", fmin
        print "convto8: fmax: ", fmax
        print "convto8: scale: ", scale
        print "convto8: shift: ", shift

    # return np.uint8( (2.**8 - 1.0) / (fmax - fmin) * (frame - fmin) )
    return cv2.convertScaleAbs(frame, alpha = scale, beta = shift)

# <codecell>

# 16 bit to 8 bit conversion

def conv16to8(frame):
    """
    convert pixel values from 16 bit to 8 bit and rescale to use the entire range of gray values
    input: frame as 16 bit image (or arbitrary accuracy)
    output: 8 bit image array (uint8)
    """
    fmin = np.double(np.nanmin(frame))
    fmax = np.double(np.nanmax(frame))
    
    return np.uint8( (2.**8 - 1.0) / (fmax - fmin) * (frame - fmin) )

# <codecell>

def align_frame(frame):
    """
    # need to transpose and fliplr, equivalent to rotate and then shift the image
    """
    return landscape(frame)

# <codecell>

def showimage(image, size = (12, 12), colorbar = False, debug = False):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if debug:
        print "plt uses backend: ", plt.get_backend()
    
    if type(size) is int:
        size = (size, size)
    
    fig, ax = plt.subplots(figsize = size)
    axim = ax.imshow(image, origin = 'lower', cmap = mpl.cm.gray)
    if colorbar:
        plt.colorbar(axim)
    plt.show()

# <codecell>

def histeq(image, number_bins = 256):
    """ Histogram equalization of a grayscale image. 
        Input:   image as numpy array
        Return:  histogram-equd image, cdf """
    imhist, bins = np.histogram(image.flatten(), number_bins, normed = True)
    cdf = imhist.cumsum() # cumulative distribution function
    
    nbits = np.int(np.around(np.log10(number_bins)))
        
    cdf = (2.0**nbits - 1.0) * cdf / cdf[-1] # normalize
    
    # linear interpolation of cdf to find nearest pixel values
    im2 = np.interp(image.flatten(), bins[:-1], cdf)
    return im2.reshape(image.shape), cdf

# <codecell>

def equalize_hist(frame):
    """
    histogram equalization: using OpenCV
    """
    return cv2.equalizeHist(convto8(frame))

# <codecell>

from scipy.signal import convolve2d

def runav2d(frame, Nwindow = 3, fillvalue = np.nan, returntype = 'float'):
    """
    simple mean blur, implemented with scipy convolve2d
    returns a float64 numpy array by default, otherwise (returntype = None) returns the same type as input image frame
    """
    
    weights = np.ones( (Nwindow, Nwindow) )
    weights /= weights.sum()
    
    c = convolve2d(frame, weights, mode = 'same', boundary = 'fill', fillvalue = fillvalue) # mode = 'full', 'valid', 'same'
    if returntype == 'float':
        return c
    else:
        return c.astype(frame.dtype)

# <codecell>

def simple_blur(frame, blocksize = 3, bordertype = cv2.BORDER_CONSTANT):
    """
    simple mean blur (running average) cv2.blur() with blocksize and borderType = CONSTANT
    the input image will be first converted to a float64 image
    """
    return cv2.blur(frame.astype(np.float64), ksize = (blocksize, blocksize), anchor = (-1, -1), borderType = bordertype)

# <codecell>

def median_blur(frame, blocksize = 3, debug = False):
    """
    median blur with blocksize
    """
    blocksize = np.int( (blocksize / 2) * 2 + 1 )
    
    if debug:
        print "median_blur: blocksize: ", blocksize
    
    if blocksize in [3, 5]:
        return cv2.medianBlur(frame, ksize = blocksize)
    else:
        return cv2.medianBlur(convto8(frame), ksize = blocksize)

# <codecell>

def gaussian_blur(frame, blocksize = 3, sigma = 1, bordertype = cv2.BORDER_CONSTANT):
    """
    Gaussian blur
    """
    blocksize = np.int( (blocksize / 2) * 2 + 1 )
    
    return cv2.GaussianBlur(frame.astype(np.float64), ksize = (blocksize, blocksize), 
                            sigmaX = sigma, sigmaY = sigma, borderType = bordertype)

# <codecell>

def threshold(frame, threshold = 0.5, normalized_threshold = True, threshold_type = cv2.THRESH_BINARY, debug = False):
    """
    tresholding an image: the input type has to be either np.uint8 or np.float32
    returns a uint8 image
    """

    fmin = np.double( np.nanmin(frame) )
    fmax = np.double( np.nanmax(frame) )
    
    if debug:
        print "fmin: ", fmin
        print "fmax: ", fmax
    
    floatframe = ( (1.0 - 0.0) / (fmax - fmin) * (np.double(frame) - fmin) ).astype(np.float32)
    
    if debug:
        print "floatframe: ", floatframe.dtype
        print "min: ", np.nanmin(floatframe)
        print "max: ", np.nanmax(floatframe)
    
    if not normalized_threshold:
        threshold = 1.0 / (fmax - fmin) * (threshold - fmin)
        if debug:
            print "normalized threshold: ", threshold
    
    retval, t = cv2.threshold(floatframe, thresh = threshold, maxval = 255, type = threshold_type)
    
    return t

# <codecell>

def adaptive_threshold(frame, blocksize = 3, offset = 0, method = 'mean'):
    """
    adaptive threshold: convert input image to an 8-bit image
    method: 'mean' or 'gaussian', default: 'mean'
    for method 'mean': threshold at point P: mean of the (blocksize X blocksize) neighborhood of point P, minus offset
    """
    blocksize = np.int( (blocksize / 2) * 2 + 1 )
    
    if method == 'mean':
        usemethod = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method == 'gaussian':
        usemethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        usemethod = cv2.ADAPTIVE_THRESH_MEAN_C
    
    return cv2.adaptiveThreshold(convto8(frame), maxValue = 2**8 - 1, blockSize = blocksize, 
                                 adaptiveMethod = usemethod, thresholdType = cv2.THRESH_BINARY, C = offset)

# <codecell>

def rotate(frame, angle_deg = 0.0, im_center = None, fillvalue = 0, interpolation = 'cubic', debug = False):
    """
    rotate image about center (default: upper left corner (0, 0)) by angle_deg degrees
    interpolate the rotated image, default method: cubic, others: linear, nearest, lanczos
    put the resulting image into an appropriately sized image (filled with fillvalue = 0) 
    and crop to minimal size
    """
    if im_center is None:
        im_center = (0, 0)
        
    if interpolation == 'cubic':
        interpolationflag = cv2.INTER_CUBIC
    elif interpolation == 'linear':
        interpolationflag = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        interpolationflag = cv2.INTER_NEAREST
    elif interpolation == 'lanczos':
        interpolationflag = cv2.INTER_LANCZOS4
    else:
        interpolationflag = cv2.INTER_CUBIC
    
    map_matrix = cv2.getRotationMatrix2D(center = im_center, angle = angle_deg, scale = 1.0) # angle in degrees

    frameheight, framewidth = frame.shape
    
    # move the image horizontally so that after rotation it won't be cut off at the left side
    dx = np.int( np.ceil( frameheight * np.sin(angle_deg / 180.0 * np.pi) ) )
    dy = np.int( np.ceil( framewidth * np.sin(angle_deg / 180.0 * np.pi) ) )
    if debug:
        print "move frame by dx = ", dx
        print "move frame by dy = ", dy
    
    if dx < 0:
        map_matrix[0, 2] += np.abs(dx)
    else:
        map_matrix[1, 2] += np.abs(dy)
    
    width = np.int( np.ceil( framewidth * np.cos(angle_deg / 180.0 * np.pi) ) + np.abs(dx) )
    height = np.int( np.ceil( frameheight * np.cos(angle_deg / 180.0 * np.pi) ) + np.abs(dy) )
    
    if debug:
        print "new width: ", width
        print "new height: ", height
    
    rotframe = cv2.warpAffine(frame, M = map_matrix, dsize = (width, height), 
                              flags = interpolationflag | cv2.cv.CV_WARP_FILL_OUTLIERS, borderValue = fillvalue)
    
    return rotframe

# <codecell>

def pyramid_down(frame, bordertype = cv2.BORDER_DEFAULT):
    """
    blurs an image and downsamples it
    seems to have a problem with BORDER_CONSTANT???
    """
    return cv2.pyrDown(frame, borderType = bordertype)

# <codecell>

def sobel(frame, xorder = 1, yorder = 0, kernelsize = 3, bordertype = cv2.BORDER_CONSTANT):
    """
    calculate the first, second, third, or mixed image derivatives using an extended Sobel operator
    """
    if kernelsize not in [1, 3, 5, 7]:
        kernelsize = 3
    
    return cv2.Sobel(frame.astype(np.float64), ddepth = -1, dx = xorder, dy = yorder, ksize = kernelsize, borderType = bordertype)

# <codecell>

def scharr(frame, xorder = 1, yorder = 0, bordertype = cv2.BORDER_CONSTANT):
    return sobel(frame, xorder = xorder, yorder = yorder, kernelsize = cv2.cv.CV_SCHARR, bordertype = bordertype)

# <codecell>

def get_structuring_element(shape = 'ellipse', blocksize = 3):
    """
    structuring elements: ellipse, rect, {star, cross}; default: ellipse
    """
    if shape == 'ellipse':
        useshape = cv2.MORPH_ELLIPSE
    elif shape == 'rect':
        useshape = cv2.MORPH_RECT
    elif shape in ['star', 'cross']:
        useshape = cv2.MORPH_CROSS
    else:
        useshape = cv2.MORPH_ELLIPSE
    
    return cv2.getStructuringElement(shape = useshape, ksize = (blocksize, blocksize), anchor = (-1, -1))

# <codecell>

def erode(frame, blocksize = 3, shape = 'ellipse', fillvalue = 0, Niterations = 1):
    """
    erosion: or 'min' operator
    bright regions are isolated and shrunk
    structuring elements: ellipse, rect, {star, cross}; default: ellipse
    """
    structuring_element = get_structuring_element(shape = shape, blocksize = blocksize)
    
    return cv2.morphologyEx(frame.astype(np.float64), op = cv2.MORPH_ERODE, kernel = structuring_element, 
                            anchor = (-1, -1), iterations = Niterations, borderType = cv2.BORDER_CONSTANT, borderValue = fillvalue)

# <codecell>

def dilate(frame, blocksize = 3, shape = 'ellipse', fillvalue = 0, Niterations = 1):
    """
    dilation: or 'max' operator
    bright regions are expanded and often joined
    structuring elements: ellipse, rect, {star, cross}; default: ellipse
    """
    structuring_element = get_structuring_element(shape = shape, blocksize = blocksize)
    
    return cv2.morphologyEx(frame.astype(np.float64), op = cv2.MORPH_DILATE, kernel = structuring_element, 
                            anchor = (-1, -1), iterations = Niterations, borderType = cv2.BORDER_CONSTANT, borderValue = fillvalue)

# <codecell>

def opening(frame, blocksize = 3, shape = 'ellipse', fillvalue = 0, Niterations = 1):
    """
    opening: first erode, then dilate
    small bright regions are removed, remaining bright regions are isolated but retain their size
    structuring elements: ellipse, rect, {star, cross}; default: ellipse
    """
    structuring_element = get_structuring_element(shape = shape, blocksize = blocksize)
    
    return cv2.morphologyEx(frame.astype(np.float64), op = cv2.MORPH_OPEN, kernel = structuring_element, 
                            anchor = (-1, -1), iterations = Niterations, borderType = cv2.BORDER_CONSTANT, borderValue = fillvalue)

# <codecell>

def closing(frame, blocksize = 3, shape = 'ellipse', fillvalue = 0, Niterations = 1):
    """
    closing: first dilate, then erode
    bright regions are joined but retain their basic size
    structuring elements: ellipse, rect, {star, cross}; default: ellipse
    """
    structuring_element = get_structuring_element(shape = shape, blocksize = blocksize)
    
    return cv2.morphologyEx(frame.astype(np.float64), op = cv2.MORPH_CLOSE, kernel = structuring_element, 
                            anchor = (-1, -1), iterations = Niterations, borderType = cv2.BORDER_CONSTANT, borderValue = fillvalue)

# <codecell>

def tophat(frame, blocksize = 3, shape = 'ellipse', fillvalue = 0, Niterations = 1):
    """
    tophat: src - opening(src)
    bright local peaks are isolated
    structuring elements: ellipse, rect, {star, cross}; default: ellipse
    """
    structuring_element = get_structuring_element(shape = shape, blocksize = blocksize)
    
    return cv2.morphologyEx(frame.astype(np.float64), op = cv2.MORPH_TOPHAT, kernel = structuring_element, 
                            anchor = (-1, -1), iterations = Niterations, borderType = cv2.BORDER_CONSTANT, borderValue = fillvalue)

# <codecell>

def blackhat(frame, blocksize = 3, shape = 'ellipse', fillvalue = 0, Niterations = 1):
    """
    blackhat: closing(src) - src
    dark holes are isolated
    structuring elements: ellipse, rect, {star, cross}; default: ellipse
    """
    structuring_element = get_structuring_element(shape = shape, blocksize = blocksize)
    
    return cv2.morphologyEx(frame.astype(np.float64), op = cv2.MORPH_BLACKHAT, kernel = structuring_element, 
                            anchor = (-1, -1), iterations = Niterations, borderType = cv2.BORDER_CONSTANT, borderValue = fillvalue)

# <codecell>

def morphgradient(frame, blocksize = 3, shape = 'ellipse', fillvalue = 0, Niterations = 1):
    """
    morphological gradient: dilate(src) - erode(src)
    bright perimeter edges are identified
    structuring elements: ellipse, rect, {star, cross}; default: ellipse
    """
    structuring_element = get_structuring_element(shape = shape, blocksize = blocksize)
    
    return cv2.morphologyEx(frame.astype(np.float64), op = cv2.MORPH_GRADIENT, kernel = structuring_element, 
                            anchor = (-1, -1), iterations = Niterations, borderType = cv2.BORDER_CONSTANT, borderValue = fillvalue)

# <codecell>

def landscape(frame):
    """
    Rotate frames, especially frames from the hemilarynx experiments,
    so that they will be in landscape mode
    Originally, the frame will have the left view at the bottom and the right view on the top
    The camera was oriented such that the top of the frames were at the downstream side, the bottom was at the upstream side of the hemilarynx
    The hemilarynx preparation was always (so far) done on the left side of the dog. The camera looked from the right side.
    So the left side in the dual views (through the prism) corresponds to the posterior vocal fold (arytenoid cartilages)
    and the right side corresponds to the anterior side of the vocal folds.
    """

    if len(frame.shape) == 2:
        return np.flipud( np.fliplr( frame.T ) )
    elif len(frame.shape) == 3:
        return np.transpose(frame, (0, 2, 1) )[:, ::-1, ::-1]
    else:
        print "landscape: data layout not understood"
        print "frame.shape: ", frame.shape
        return frame

# <codecell>

def ratioTest(matches, ratio = 0.65):
    """
    ratio test of matches, default ratio = 0.65
    """
    removed = 0
    
    clean_matches = []
    
    for match in matches:
        if len(match) > 1:
            # match[0].distance is always smaller (better match) than match[1].distance
            if (match[0].distance / match[1].distance) > ratio:
                removed += 1
            else:
                clean_matches.append(match)
        else:
            removed += 1
            
    return removed, clean_matches

# <codecell>

def blobdetect(frame):
    """
    The class implements a simple algorithm for extracting blobs from an image:

    * Convert the source image to binary images by applying thresholding with several thresholds from 
        minThreshold (inclusive) to maxThreshold (exclusive) with distance thresholdStep between neighboring thresholds.
    * Extract connected components from every binary image by findContours() and calculate their centers.
    * Group centers from several binary images by their coordinates. Close centers form one group that 
        corresponds to one blob, which is controlled by the minDistBetweenBlobs parameter.
    * From the groups, estimate final centers of blobs and their radiuses and return as locations and sizes of keypoints.
    """
    
    params = cv2.SimpleBlobDetector_Params()
    
    # detect dark blobs, so filter by color
    params.filterByColor = True
    params.blobColor = 0 # detect dark blob, for white: 255
    
    params.filterByArea = True
    params.minArea = 6
    params.maxArea = 100
    
    params.minThreshold = 1
    params.maxThreshold = 200
    
    params.minDistBetweenBlobs = 10
    
    detector = cv2.SimpleBlobDetector(params)
    
    # due to thresholding the input image needs to either 8 bit or 32 bit floating point (both types?)???
    # ONLY uint8 works: although threshold can handle float32, findcontours can NOT handle float32
    # so the common denominator should be uint8
    points_list = detector.detect( convto8( frame ) )
    
    return points_list
    
    # sort points with respect to x coordinate
    points_sorted = sorted(points_list, key = lambda x:x.pt[0])
    
    kp_x = [kp.pt[0] for kp in pts_sorted]
    kp_y = [kp.pt[1] for kp in pts_sorted]

# <codecell>

def find_closest_keypoint(point, keypoints):
    """
    find the keypoint closest to the given point
    returns: index_minimum_keypoints, dx, dy, distance, closest_point
    """
    
    dx = point[0] - keypoints[0]
    dy = point[1] - keypoints[1]
        
    ds = np.hypot(dx, dy)
        
    minind = np.argmin(ds)
    
    mindx = dx[minind]
    mindy = dy[minind]
    mindist = ds[minind]
    closest_point = keypoints[:, minind]
    
    return minind, mindx, mindy, mindist, closest_point

