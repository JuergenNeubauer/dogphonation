# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Hemilarynx analysis
# ===================
# 
# hemilarynx data directories
# ---------------------------
# 
# * InVivoDog_2012_02_15
# * InVivoDog_2012_09_26
# * InVivoDog_2012_10_10
#     * ventricular folds
# * InVivoDog_2012_10_17
#     * has PCA stimulation

# <markdowncell>

# 9/26/12
# =======
# 
# left RLN:
# --------
# * Left RLN Range Checking:  Good
# * Left RLN Range finding: Good quality; significant scrunch at end 
# * Left Range Rechecking: lose some posterior markers in one prism view
# 
# left SLN:
# --------
# * leftSLN range-checking 3:53-3:54: excellent quality
# * left SLN-range finding 3:34-3:38: excellent quality; thyroid cartilage continues to move even after vocal fold elongation is complete
# * leftSLN range_rechecking 4:24-4:24: some loss of quality in posterior prism view
# 
# left SLN-left RLN_alternating
# ----------------------------
# * Good quality, last row of markers disappears as it scrunches excessively
# 
# Left SLN-left RLN_alternating phonation
# ----------------------------------------
# * Good quality;  vibration?
# * Need to check the audio files

# <markdowncell>

# 10/10/12
# =========
# 
# Left RLN-range finding:
# ------------------------
# * excellent: can really see difference without TA (RLN was LCA)
# 
# Left SLN range finding: 
# -----------------------
# * excellent: may be better than 9/26/12; this day had FVC intact
# 
# Left TA_range finding: 
# ----------------------
# * excellent view, BUT no scrunch so ? TA
# 
# RLN
# ---
# 
# SLN
# ---
# 
# SLN-TA
# ------
# 
# SLN-trunk RLN: 
# --------------
# * not as good
# 
# TA
# --
# 
# TA-trunkRLN
# ------------

# <markdowncell>

# 10/17/13
# ========
# 
# * Much better TA, LCA, PCA, CT behavior
# * Smudge in mid upper membranous vocal fold

# <codecell>

import sys, glob, os
import numpy as np

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib as mpl
import matplotlib.pyplot as plt

# <codecell>

mpl.interactive(False)
plt.interactive(False)

print mpl.is_interactive()
print plt.isinteractive()

# <codecell>

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['image.cmap'] = 'jet'

# make plots in Notebook smaller, so they would also take less memory and storage size
mpl.rcParams['savefig.dpi'] = 100

sys.path.append("/extra/InVivoDog/python/cine/tools")

from dogdata import DogData
from cine import Cine

import xlrd

# <codecell>

import cv2

# <codecell>

# similar to import but inserts all declarations into the namespace of this notebook
%run "tools for hemilarynx analysis.py"

# <codecell>

ls -alot ../InVivoDog_2012_09_26/

# <codecell>

ls -alot ../InVivoDog_2012_10_17/

# <codecell>

ls -alot ../InVivoDog_2012_10_17/calibration/

# <codecell>

hemi_dir = '/extra/InVivoDog/InVivoDog_2012_10_17/'

calibration_dir = '/extra/InVivoDog/InVivoDog_2012_10_17/calibration/'

calibration = Cine(initialdir = calibration_dir, 
                   cinefilename = 'calibration_000_Wed Oct 17 2012 14 13 10.094 084.001.cine', 
                   debug = True)

# <codecell>

frame_calib = landscape( calibration.getframe(0) ) [0]

# mpl.rcParams = mpl.rcParamsDefault

showimage(frame_calib)

# <codecell>

rotate_frame_calib = -1 # angle in degrees

frame_calib_r = rotate(frame_calib, angle_deg = rotate_frame_calib)

showimage(frame_calib_r)

# <codecell>

frame_calib = frame_calib_r

# <codecell>

x_split = 335
x_split_frame_calib = x_split

# dict comprehension INCOMPLETE
# {side: frame_calib[:, ] for side in ['left', 'right']}

calib = dict(left = frame_calib_r[:, :x_split],
             right = frame_calib_r[:, x_split:])

print "calib left: ", calib['left'].shape
print "calib right: ", calib['right'].shape

showimage(calib['left'], size = 7)

# <codecell>

morphframe = dilate(frame_calib, blocksize = 4)

showimage(morphframe, size = 10)

# <codecell>

# blurframe = simple_blur(calib['left'], blocksize = 50, bordertype = cv2.BORDER_CONSTANT)
# blurframe = median_blur(calib['left'], blocksize = 5)
blurframe = gaussian_blur(calib['left'], blocksize = 50, sigma = 3)

print "calib: ", calib['left'].dtype
print "blurframe: ", blurframe.dtype
print blurframe.min()
print blurframe.max()

if blurframe.dtype == np.uint8:
    diff_frame = convto8(calib['left']) - blurframe
else:
    diff_frame = calib['left'] - blurframe
    
print "diff_frame: ", diff_frame.dtype
print diff_frame.min()
print diff_frame.max()

# showimage(blurframe, size = 6)
# showimage(diff_frame, size = 6)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(blurframe, cmap = mpl.cm.gray)
ax[1].imshow(diff_frame, cmap = mpl.cm.gray)

plt.show()

# <codecell>

framedata = calib['left']
framedata = blurframe
# framedata = diff_frame
# framedata = morphframe

offset = -15
blocksize = 10

abin = adaptive_threshold(framedata, blocksize = blocksize, offset = offset)

showimage(abin, size = 7, colorbar = True)

# <codecell>

showimage(equalize_hist(framedata), size = 6)

# <codecell>

fig, ax = plt.subplots(1, 2, figsize = (15, 20))

ax[0].imshow(calib['left'], cmap = mpl.cm.gray)

ax[1].imshow(calib['right'], cmap = mpl.cm.gray)
ax[1].set_yticklabels([])

plt.show()

# <codecell>

Nwindow = 30

# filt_calib = {side: runav2d(calib[side], Nwindow) for side in ['left', 'right']}
filt_calib = {side: simple_blur(calib[side], blocksize = Nwindow) for side in ['left', 'right']}

clean_calib = {side: calib[side] - filt_calib[side] for side in ['left', 'right']}

inv_clean_calib = {side: (2**8 - 1) - convto8(clean_calib[side]) for side in ['left', 'right']}

# <codecell>

print "calib: ", calib['left'].dtype
print np.nanmin(calib['left'])
print np.nanmax(calib['left'])

fl = filt_calib['left']
print "filt_calib: ", fl.dtype
print np.nanmin(fl)
print np.nanmax(fl)

cc = clean_calib['left']
print "clean_calib: ", cc.dtype
print np.nanmin(cc)
print np.nanmax(cc)

# <codecell>

fig, ax = plt.subplots(1, 2, figsize = (15, 20))

im0 = ax[0].imshow(filt_calib['left'], cmap = mpl.cm.gray)
# plt.colorbar(im0, cax = ax[0])

ax[1].imshow(convto8(clean_calib['left']), cmap = mpl.cm.gray)
# plt.colorbar()
ax[1].set_yticklabels([])

plt.show()

# <codecell>

fig, ax = plt.subplots(1, 2, figsize = (15, 20))

ax[0].imshow(calib['right'], cmap = mpl.cm.gray)

ax[1].imshow(clean_calib['right'], cmap = mpl.cm.gray)
# plt.colorbar()
ax[1].set_yticklabels([])

plt.show()

# <codecell>

fig, ax = plt.subplots(1, 2, figsize = (15, 20))

ax[0].imshow(inv_clean_calib['left'], cmap = mpl.cm.gray)
# plt.colorbar()
ax[0].grid(False)

ax[1].imshow(inv_clean_calib['right'], cmap = mpl.cm.gray)
# plt.colorbar()
ax[1].grid(False)

plt.show()

# <rawcell>

# # import PreprocessingCalibrationTarget
# # detect_local_minima = PreprocessingCalibrationTarget.detect_local_minima
# 
# # locmin = detect_local_minima(inv_clean_calib['left'])

# <codecell>

surf = cv2.SURF(# hessianThreshold = 400.0,
                # nOctaves = 4,
                # nOctaveLayers = 3,
                )

for name in surf.getParams():
    print "SURF: %s: " % name, getattr(surf, name)

# <markdowncell>

# SIFT parameters:
# ================
# 
# * nfeatures – The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
# 
# * nOctaveLayers – The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
# 
# * contrastThreshold – The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
# 
# * edgeThreshold – The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
# 
# * sigma – The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.

# <codecell>

sift = cv2.SIFT(# nfeatures = 300,
                # sigma = 2.0,
                # contrastThreshold = 0.2,
                # edgeThreshold = 1.5,
                # nOctaveLayers = 3,
                )

for name in sift.getParams():
    try:
        print "SIFT: %s: "% name, getattr(sift, name)
    except:
        print
    
detector = sift

# <rawcell>

# # http://jayrambhia.wordpress.com/2013/01/18/sift-keypoint-matching-using-python-opencv/
# 
# fd = cv2.FeatureDetector_create('SIFT')
# de = cv2.DescriptorExtractor_create('SIFT')
# 
# skp = fd.detect(img)
# skp, sd = de.compute(img, skp)
# 
# flann_params = dict(algorithm = 1, trees = 4)
# flann = cv2.flann_Index(sd, flann_params)
# idx, dist = flann.knnSearch(td, 1, params={})

# <rawcell>

# # http://code.opencv.org/projects/opencv/repository/revisions/master/entry/samples/python2/plane_tracker.py
# FLANN_INDEX_KDTREE = 1
# 
# FLANN_INDEX_LSH    = 6
# 
# flann_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
# 
# detector = cv2.ORB( nfeatures = 1000 )
# matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
# 
# # discussion of ORB
# # http://stackoverflow.com/questions/7232651/how-does-opencv-orb-feature-detector-work
# # http://www.willowgarage.com/sites/default/files/orb_final.pdf

# <rawcell>

# # sift.detectAndCompute(image, mask, descriptors, useProvidedKeyPoints)

# <codecell>

keypoints = dict()
descriptors = dict()

for do_side in ['left', 'right']:
    work_on_image = calib[do_side]
    # work_on_image = clean_calib[do_side])
    # work_on_image = inv_clean_calib[do_side]

    surf_kp, descr = detector.detectAndCompute(convto8(work_on_image), mask = None)
    print "%s: " % do_side, len(surf_kp)
    
    keypoints[do_side] = surf_kp
    descriptors[do_side] = descr

# <codecell>

skp_x = np.array([kp.pt[0] for kp in surf_kp])
skp_y = np.array([kp.pt[1] for kp in surf_kp])
skp_resp = np.array([kp.response for kp in surf_kp])

p_y, ind_p_y = np.unique(skp_y, return_index = True)
p_x = skp_x[ind_p_y]
len(p_y)

# <codecell>

if False:
    plt.plot(sorted([kp.pt for kp in surf_kp], key = lambda x: x[0]))
    plt.show()

# <codecell>

plt.subplots(1, 1, figsize = (12, 12))

plt.imshow(work_on_image, cmap = mpl.cm.gray)

plt.plot(skp_x, skp_y, 'ro', mfc = 'None', mec = 'red', ms = 20)
# plt.plot(p_x, p_y, '*', mfc = 'None', mec = 'green', ms = 20)

plt.grid(True)

plt.xlim(xmin = 0) #, xmax = 325)
plt.ylim(ymax = 0) #, ymax = 375)

plt.show()

# <headingcell level=1>

# Feature Matching

# <rawcell>

# # not needed
# matcher = cv2.DescriptorMatcher_create('BruteForce')
# for param in matcher.getParams():
#     try:
#         print "%s: " % param, getattr(matcher, param)
#     except:
#         print

# <codecell>

matcher = cv2.BFMatcher(normType = cv2.NORM_L2,
                        crossCheck = False # False: needed for knnMatch
                        )

for param in matcher.getParams():
    try:
        print "%s: " % param, getattr(matcher, param)
    except:
        print

# <codecell>

matches = matcher.match(descriptors['left'], descriptors['right'])
len(matches)

# <codecell>

# the BFMatcher must NOT have crossCheck = True
matches1 = matcher.knnMatch(descriptors['left'], trainDescriptors = descriptors['right'], k = 2)
matches2 = matcher.knnMatch(descriptors['right'], trainDescriptors = descriptors['left'], k = 2)

print "matches1: ", len(matches1)
print "matches2: ", len(matches2)

# <codecell>

removed, clean_matches1 = ratioTest(matches1) # , ratio = 0.8)
print removed

# <codecell>

removed, clean_matches2 = ratioTest(matches2) # , ratio = 0.8)
print removed

# <codecell>

matches = [m[0] for m in clean_matches1]

# <codecell>

sortind = np.argsort([m.distance for m in matches])

# take the strongest matches, say the strongest 100
strong_matches = [matches[k] for k in sortind[:100:1]]

matched_keypoints_Idx = dict()
matched_keypoints_Idx['left'] = [m.queryIdx for m in strong_matches]
matched_keypoints_Idx['right'] = [m.trainIdx for m in strong_matches]

# <codecell>

matched_keypoints = dict()
for side in ['left', 'right']:
    matched_keypoints[side] = [keypoints[side][m] for m in matched_keypoints_Idx[side]]

# <codecell>

skp_x = dict()
skp_y = dict()

for do_side in ['left', 'right']:

    surf_kp = matched_keypoints[do_side]

    skp_x[do_side] = np.array([kp.pt[0] for kp in surf_kp])
    skp_y[do_side] = np.array([kp.pt[1] for kp in surf_kp])
    skp_resp = np.array([kp.response for kp in surf_kp])

    p_y, ind_p_y = np.unique(skp_y[do_side], return_index = True)
    p_x = skp_x[do_side][ind_p_y]
    print "unique: %s: " % do_side, len(p_y)

# <codecell>

allx = np.vstack((skp_x['left'], x_split + skp_x['right']))
ally = np.vstack((skp_y['left'], skp_y['right']))

# <codecell>

plt.plot(allx, ally, '-')
plt.show()

# <codecell>

calib_both = np.hstack((calib['left'], calib['right']))
calib_both.shape

# <codecell>

plt.subplots(1, 1, figsize = (15, 15))

plt.imshow(convto8(calib_both), cmap = mpl.cm.gray)

plt.plot(allx, ally, '-')
# plt.plot(skp_x, skp_y, 'ro', mfc = 'None', mec = 'red', ms = 20)
# plt.plot(p_x, p_y, '*', mfc = 'None', mec = 'green', ms = 20)

plt.grid(True)

# plt.xlim(xmin = 0, xmax = 325)
# plt.ylim(ymin = 0, ymax = 375)

plt.show()

# <headingcell level=1>

# Blob detection

# <markdowncell>

# The class implements a simple algorithm for extracting blobs from an image:
# 
# * Convert the source image to binary images by applying thresholding with several thresholds from minThreshold (inclusive) to maxThreshold (exclusive) with distance thresholdStep between neighboring thresholds.
# * Extract connected components from every binary image by findContours() and calculate their centers.
# * Group centers from several binary images by their coordinates. Close centers form one group that corresponds to one blob, which is controlled by the minDistBetweenBlobs parameter.
# * From the groups, estimate final centers of blobs and their radiuses and return as locations and sizes of keypoints.

# <codecell>

def blobdetect(frame):
    """
    """
    
    try:
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
    except:
        return
    
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

plt.plot( 255 - (inv_clean_calib['left'])[45:55, :].T)
plt.show()

# <codecell>

# showimage( 255 - convto8(calib['left']), size = 6)
showimage( opening(inv_clean_calib['left'], blocksize = 7), size = 6)
# showimage( 255 - convto8(dilate(calib['left'], blocksize = 4)), size = 6)

# <codecell>

showimage( 255 - ( opening(clean_calib['right'], blocksize = 3, Niterations = 2) ), size = 7)

# <codecell>

pts_list = dict()

for side in ['left', 'right']:
    # detect_image = 255 - convto8( dilate(calib[side], blocksize = 4) )
    
    detect_image = 255 - convto8( opening(clean_calib[side], blocksize = 3, Niterations = 2) )
    
    # pts_list[side] = detector.detect( convto8( detect_image ) ) # inv_clean_calib[side]) # mask = None
    pts_list[side] = blobdetect(detect_image)
    
print len(pts_list['left'])
print len(pts_list['right'])

# <codecell>

pts_sorted = dict()
kp_coord = dict()
kp_x = dict()
kp_y = dict()
kp_all = dict()

for showside in ['left', 'right']:
    # sort points with respect to x coordinate
    pts_sorted[showside] = sorted(pts_list[showside], key = lambda x:x.pt[0])

    kp_coord[showside] = [kp.pt for kp in pts_sorted[showside]]
    
    kp_x[showside] = [kp.pt[0] for kp in pts_sorted[showside]]
    kp_y[showside] = [kp.pt[1] for kp in pts_sorted[showside]]
    
    kp_all[showside] = np.array([kp_x[showside], kp_y[showside]])

# <codecell>

outfilename = "dataexchange.npz"

savez_dict = dict(calib = calib, kp_coord = kp_coord) # kp_x = kp_x, kp_y = kp_y)

np.savez_compressed(outfilename, **savez_dict)

# <codecell>

for side in calib:
    ymax, xmax = calib[side].shape
    print side, ': xmax = ', xmax, ', ymax = ', ymax

# <codecell>

showside = 'left'
# showside = 'right'

# <codecell>

fig, ax = plt.subplots(1, 2, figsize = (20, 20))

for num, showside in enumerate(['left', 'right']):
    ax[num].imshow(calib[showside], cmap = mpl.cm.gray)
    
    ax[num].plot(kp_x[showside], kp_y[showside], 'ro', mfc = 'None', mec = 'red', ms = 20, alpha = 0.5)
    ax[num].plot(kp_x[showside], kp_y[showside], 'gx', ms = 7, alpha = 0.5)

for a in ax:
    a.set_xlim(xmin = 0) #, xmax = 325)
    a.set_ylim(ymin = 0) #, ymax = 375)

    
fig.savefig('calibrationspoints.png', orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
            
plt.show()

# <rawcell>

# help cv2.findCirclesGrid
# 
# cv2.findCirclesGrid(inv_clean_calib['left'], (8, 17), flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING) #, blobDetector = detector)

# <codecell>

%%file manually_remove_points.py

import numpy as np

# <codecell>

%%file order_points.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# <codecell>

%%file interactive_data.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

outfilename = "dataexchange.npz"

data = np.load(outfilename)

calib = data['calib'].tolist()
kp_coord = data['kp_coord'].tolist()

def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)

    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x: %1.4f, y: %1.4f, z: %1.4f'%(x, y, z)
    else:
        return 'x: %1.4f, y: %1.4f'%(x, y)

clickpoints = dict()
clicked_kp = dict()

for showside in ['left', 'right']:
    kp_x = [kp[0] for kp in kp_coord[showside]]
    kp_y = [kp[1] for kp in kp_coord[showside]]
    
    fig, ax = plt.subplots(figsize = (12, 12))
    fig.suptitle('calibration: %s\np10, p1X, p1Y, p20, p2X, p2Y' % showside)
    fig.canvas.manager.set_window_title('calibration: %s' % showside)
    
    ax.imshow( calib[showside], cmap = mpl.cm.gray, zorder = 1)
    
    ax.plot(kp_x, kp_y, 'ro', mfc = 'None', mec = 'red', ms = 20, zorder = 10)
    ax.plot(kp_x, kp_y, 'gx', ms = 7, zorder = 20)
    
    plt.grid(True)
    
    plt.xlim(xmin = 0) #, xmax = 325)
    plt.ylim(ymin = 0) #, ymax = 375)
    
    X = calib[showside]
    
    numrows, numcols = X.shape
    
    ax.format_coord = format_coord
    
    clickpoints[showside] = fig.ginput(n = 6, timeout = -1, show_clicks = True)
    
    kp_all = np.array([kp_x, kp_y])
    
    # find the keypoints closest to the clicked points
    kp_indices = []
    for pt in clickpoints[showside]:
        dx = pt[0] - kp_all[0]
        dy = pt[1] - kp_all[1]
        
        ds = np.hypot(dx, dy)
        
        kp_indices.append(np.argmin(ds))
    
    clicked_kp[showside] = kp_all[:, kp_indices]
    
    ax.plot(clicked_kp[showside][0, :], clicked_kp[showside][1, :], 'yo', mfc = 'None', mec = 'yellow', ms = 20, zorder = 20)
    
    # print the clickpoints so that the calling routine can parse the stdout of this function to get the clickpoints
    ### Will be done from the calling routine!!!
    # print "clickpoints: ", clickpoints
    
    p10, p1X, p1Y, p20, p2X, p2Y = clicked_kp[showside].T # [np.array(p).T for p in clickpoints]
    
    # direction vectors of x- and y-axes for the first set of points on plane 0 (closest to the hypothenuse face of the prism)
    e1_X = p1X - p10
    e1_Y = p1Y - p10
    
    # direction vectors for the second set of points on plane 1 of the calibration target
    e2_X = p2X - p20
    e2_Y = p2Y - p20
    
    # to display the direction show the directions a times its length
    a = 4.0
    
    l1_X = np.array([p10, p10 + a * e1_X]).T
    l1_Y = np.array([p10, p10 + a * e1_Y]).T
    
    l2_X = np.array([p20, p20 + a * e2_X]).T
    l2_Y = np.array([p20, p20 + a * e2_Y]).T
    
    ax.plot(l1_X[0, :], l1_X[1, :], 'g.-', zorder = 1000)
    ax.plot(l1_Y[0, :], l1_Y[1, :], 'b.-', zorder = 1000)
    
    ax.plot(l2_X[0, :], l2_X[1, :], 'y.-', zorder = 1000)
    ax.plot(l2_Y[0, :], l2_Y[1, :], 'm.-', zorder = 1000)
    
    fig.canvas.draw()
    
# start the event loop so the plots don't disappear and the program waits, blocks
plt.show()

# <codecell>

%%script python --bg --err stderror_pipe --out stdout_pipe
import sys

print sys.argv
print 'Python version: %s' % sys.version

import matplotlib as mpl
import matplotlib.pyplot as plt

print "mpl: ", mpl.get_backend()

import interactive_data

# create output that can be parsed from the output pipe to retrieve the data
for showside in ['left', 'right']:
    print "%s: clickpoints: " % showside, interactive_data.clickpoints[showside]
    print "%s: clicked_kp: " % showside, interactive_data.clicked_kp[showside].tolist()

# <rawcell>

# # %%script python --bg --err stderror_pipe --out stdout_pipe

# <codecell>

if not stderror_pipe.closed:
    for stderrorline in stderror_pipe.readlines():
        print stderrorline
    stderror_pipe.close()
else:
    print "stderror_pipe closed: no more data"

# <codecell>

if stdout_pipe.closed:
    print "stdout_pipe closed: no more data"
else:
    clickpoints = dict()
    clicked_kp = dict()
    stdout_save = []
    
    for stdoutline in stdout_pipe.readlines():
        stdout_save.append(stdoutline)
        print stdoutline
    
        for side in ['left', 'right']:
            if stdoutline.startswith('%s: clickpoints:' % side):
                clickpoints[side] = eval( stdoutline.replace('%s: clickpoints:' % side, '') )
            if stdoutline.startswith('%s: clicked_kp:' % side):
                clicked_kp[side] = eval( stdoutline.replace('%s: clicked_kp:' % side, '') )
            
    stdout_pipe.close()
    # make the lists into numpy arrays            
    for side in ['left', 'right']:
        clicked_kp[side] = np.array(clicked_kp[side])

# <codecell>

np.savez('clicked_keypoints.npz', clicked_kp = clicked_kp)

# <codecell>

clicked_kp = np.load('clicked_keypoints.npz')['clicked_kp'].tolist()

# <codecell>

showside = 'right' # 'left'

fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('calibration: %s' % showside)
    
ax.imshow( calib[showside], cmap = mpl.cm.gray, zorder = 1)
    
ax.plot(kp_x[showside], kp_y[showside], 'ro', mfc = 'None', mec = 'red', ms = 20, zorder = 10)
ax.plot(kp_x[showside], kp_y[showside], 'gx', ms = 7, zorder = 20)
    
plt.grid(True)
    
plt.xlim(xmin = 0) #, xmax = 325)
plt.ylim(ymin = 0) #, ymax = 375)

ax.plot(clicked_kp[showside][0, :], clicked_kp[showside][1, :], 'yo', mfc = 'None', mec = 'yellow', ms = 20, zorder = 20)
    
p10, p1X, p1Y, p20, p2X, p2Y = clicked_kp[showside].T

# direction vectors of x- and y-axes for the first set of points on plane 0 (closest to the hypothenuse face of the prism)
e1_X = p1X - p10
e1_Y = p1Y - p10

# direction vectors for the second set of points on plane 1 of the calibration target
e2_X = p2X - p20
e2_Y = p2Y - p20

# to display the direction show the directions a times its length
a = 4.0
    
l1_X = np.array([p10, p10 + a * e1_X, p10 - a * e1_X]).T
l1_Y = np.array([p10, p10 + a * e1_Y, p10 - a * e1_Y]).T
    
l2_X = np.array([p20, p20 + a * e2_X, p20 - a * e2_X]).T
l2_Y = np.array([p20, p20 + a * e2_Y, p20 - a * e2_Y]).T
    
ax.plot(l1_X[0, :], l1_X[1, :], 'm.-', zorder = 1000)
ax.plot(l1_Y[0, :], l1_Y[1, :], 'm.-', zorder = 1000)
    
ax.plot(l2_X[0, :], l2_X[1, :], 'b.-', zorder = 1000)
ax.plot(l2_Y[0, :], l2_Y[1, :], 'b.-', zorder = 1000)
    
ax.plot(np.array(point_X)[:, 0], np.array(point_X)[:, 1], 'go', mfc = 'None', mec = 'green', ms = 20, zorder = 50)
    
fig.canvas.draw()

plt.show()

# <codecell>

showside = 'right'

p10, p1X, p1Y, p20, p2X, p2Y = clicked_kp[showside].T

# direction vectors of x- and y-axes for the first set of points on plane 1 (closest to the hypothenuse face of the prism)
e1_X = p1X - p10
e1_Y = p1Y - p10

# direction vectors for the second set of points on plane 2 of the calibration target
e2_X = p2X - p20
e2_Y = p2Y - p20

# coordinates in units of e1_X and e1_Y (or e2_X and e2_Y, respectively)
point_coord = [-2, 1]

point = p10 + point_coord[0] * e1_X + point_coord[1] * e1_Y
# point = p20 + point_coord[0] * e2_X + point_coord[1] * e2_Y

find_closest_keypoint(point, kp_all[showside])

# <codecell>

print e1_X
print e1_Y
print e2_X
print e2_Y

# length of direction vector in pixels
print "|e1_X| = ", np.sqrt(np.sum(e1_X**2))
print "|e1_Y| = ", np.sqrt(np.sum(e1_Y**2))
print "|e2_X| = ", np.sqrt(np.sum(e2_X**2))
print "|e2_Y| = ", np.sqrt(np.sum(e2_Y**2))

# <codecell>

calibrationpoints = {'left': {0: None, 1: None},
                     'right': {0: None, 1: None}}

# <codecell>

showside = ['left', 'right'][1]

plane = [0, 1][0]

ymax, xmax = calib[showside].shape
xmax -= 1
ymax -= 1
    
p10, p1X, p1Y, p20, p2X, p2Y = clicked_kp[showside].T

# direction vectors of x- and y-axes for the first set of points on plane 0 (closest to the hypothenuse face of the prism)
e1_X = p1X - p10
e1_Y = p1Y - p10
# direction vectors for the second set of points on plane 1 of the calibration target
e2_X = p2X - p20
e2_Y = p2Y - p20

# length of direction vector in pixels: norm
l_e1_X = np.sqrt(np.sum(e1_X**2))
l_e1_Y = np.sqrt(np.sum(e1_Y**2))
l_e2_X = np.sqrt(np.sum(e2_X**2))
l_e2_Y = np.sqrt(np.sum(e2_Y**2))

index_X = []
point_X = []

x0 = 0
y0 = 0
dx = 1
dy = 1

ind_x = x0
ind_y = y0

if plane == 0:
    p0 = p10
    eX = e1_X
    eY = e1_Y
    lX = l_e1_X
    lY = l_e1_Y
if plane == 1:
    p0 = p20
    eX = e2_X
    eY = e2_Y
    lX = l_e2_X
    lY = l_e2_Y
    
count = 0
nextline = 0
    
while True:
    count += 1
    # coordinates in units of e1_X and e1_Y (or e2_X and e2_Y, respectively)
    point_coord = [ind_x, ind_y, plane]
    point = p0 + point_coord[0] * eX + point_coord[1] * eY
    
    print point_coord
    
    # prevent an infinite while loop
    if count > 1000:
        break
    
    if point[0] > xmax:
        # reached end of row marching right, go left from the origin
        dx = -1
        ind_x = x0 + dx
        nextline = 0
        continue
    elif point[0] < 0:
        # reached end of row marching left, go to next row in vertical direction
        ind_x = x0
        ind_y += dy
        dx = 1
        nextline = 1
        continue
    else:
        ind_x += dx
        
    if point[1] > ymax and np.abs(ind_x) == 1:
        # reached end of column going upwards, go downwards from the origin
        ind_x = x0
        dx = 1
        dy = -1
        ind_y = y0 + dy
        nextline = 1
        continue
    elif point[1] < 0:
        # reached end of column going downwards, stop now?
        # we might miss points if they are along inclined lines
        break
        
    index_minimum_keypoint, dxp, dyp, distance, closest_point = find_closest_keypoint(point, kp_all[showside])
    print point_coord, distance, dxp, dyp
    
    if nextline == 0:
        if distance < lX * 0.15:
            index_X.append(point_coord)
            point_X.append(closest_point)
            print 'point added'
    elif nextline == 1:
        nextline = 0
        if distance < lY * 0.07:
            index_X.append(point_coord)
            point_X.append(closest_point)
            print 'point added'
            
    if len(index_X) > kp_all[showside].shape[: 1]:
        break

# <codecell>

calibrationpoints[showside][plane] = dict(point_X = np.array(point_X),
                                          index_X = np.array(index_X))

# <codecell>

# DaVis calibration target dimensions in mm
DaVis_calibrationtarget = dict(dx =  2.0, # left-right in image; posterior-anterior wrt larynx
                               dy =  2.0, # down-up in image; inferior-superior wrt larynx
                               dz = -0.5 * 0) # out of plane towards camera; lateral to medial wrt larynx

scaling_mm = np.array([DaVis_calibrationtarget[item] for item in sorted(DaVis_calibrationtarget.keys())])

# <codecell>

objectpoints = dict()
imagepoints = dict()

# <codecell>

all_cameramatrix = dict()
all_dist_coefs = dict()

# <codecell>

side = ['left', 'right'][1]

objectpoints.update({side: [(calibrationpoints[side][0]['index_X'] * scaling_mm).astype(np.float32), 
                            (calibrationpoints[side][1]['index_X'] * scaling_mm).astype(np.float32)]})

imagepoints.update({side: [calibrationpoints[side][0]['point_X'].astype(np.float32),
                           calibrationpoints[side][1]['point_X'].astype(np.float32)]})

# <codecell>

image_height, image_width = calib[side].shape

# <codecell>

# estimates the object pose given a set of object points, their corresponding image projections, 
# as well as the camera matrix and the distortion coefficients

# cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]])
# -> retval, rvec, tvec

w, h = image_width, image_height

dist_coefs = np.zeros(4)

fx = 200 # 200 mm lens as initial guess

cameramatrix = np.float64([[fx*w,  0,     0.5*(w-1)],
                           [0.0,   fx*w,  0.5*(h-1)],
                           [0.0,   0.0,   1.0]])

retval, rvec, tvec = cv2.solvePnP(objectPoints = objectpoints[side][0], 
                                  imagePoints = imagepoints[side][0], 
                                  cameraMatrix = cameramatrix, 
                                  distCoeffs = dist_coefs,
                                  # flags = cv2.CV_ITERATIVE
                                  )

# <codecell>

# calibrateCamera(objectPoints, imagePoints, imageSize[, cameraMatrix[, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]]]) 
# -> retval, cameraMatrix, distCoeffs, rvecs, tvecs

allflags = [# cv2.CALIB_USE_INTRINSIC_GUESS, 
            # cv2.CALIB_FIX_PRINCIPAL_POINT,
            # cv2.CALIB_FIX_ASPECT_RATIO,
            # cv2.CALIB_FIX_K2,                # corresponding radial distortion coefficient is not changed
            # cv2.CALIB_FIX_K3,                # should be fixed unless fish-eye lens used; usually very small
            cv2.CALIB_RATIONAL_MODEL,      # Coefficients k4, k5, and k6 are enabled
            # cv2.CALIB_ZERO_TANGENT_DIST,     # Tangential distortion coefficients (p1, p2) are set to zeros and stay zero
            ]

w, h = image_width, image_height

dist_coefs = np.zeros(8) # k1, k2, p1, p2, k3, k4, k5, k6

f = 200 # 200 mm lens as initial guess

# A = [[fx, 0, cx],
#      [0, fy, cy],
#      [0, 0,  1]]

fx = f * w
fy = f * w
cx = 0.5 * w
cy = 0.5 * h
cameramatrix = np.float64([[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,  1]])

rms, cameramatrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objectPoints = objectpoints[side], 
                                                                   imagePoints = imagepoints[side], 
                                                                   imageSize = (image_width, image_height),
                                                                   cameraMatrix = cameramatrix,
                                                                   distCoeffs = dist_coefs,
                                                                   flags = sum(allflags))
print "reprojection error [pixels]: ", rms

# <codecell>

all_cameramatrix[side] = cameramatrix
all_dist_coefs[side] = dist_coefs

# <codecell>

cameramatrix

# <codecell>

rvecs # rotation vectors (see Rodrigues() ) estimated for each pattern view

# <codecell>

tvecs # translation vectors estimated for each pattern view

# <codecell>

zip(['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'], dist_coefs)

# <codecell>

undistored_calib = cv2.undistort(src = calib[side], cameraMatrix = cameramatrix, distCoeffs = dist_coefs)

showimage(undistored_calib, size = 7)

# <rawcell>

# calibrationMatrixValues
# Parameters:	
# cameraMatrix – Input camera matrix that can be estimated by calibrateCamera() or stereoCalibrate() .
# imageSize – Input image size in pixels.
# apertureWidth – Physical width of the sensor.
# apertureHeight – Physical height of the sensor.
# fovx – Output field of view in degrees along the horizontal sensor axis.
# fovy – Output field of view in degrees along the vertical sensor axis.
# focalLength – Focal length of the lens in mm.
# principalPoint – Principal point in pixels.
# aspectRatio – fy/fx

# <codecell>

# useful camera characteristics from the camera matrix
cv2.calibrationMatrixValues(cameraMatrix = cameramatrix, imageSize = (image_width, image_height), 
                            apertureWidth = , apertureHeight = )

# <codecell>

# initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) -> map1, map2

a_deg = 0 # angle in degrees
a = a_deg * np.pi / 180.0

R = np.float64([[np.cos(a), np.sin(a), 0],
                [-np.sin(a), np.cos(a), 0],
                [0, 0, 1]])

map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix = cameramatrix, distCoeffs = dist_coefs, 
                                         R = R, newCameraMatrix = cameramatrix,
                                         size = (1 * image_width, 1 * image_height), m1type = cv2.CV_32FC1)

# <codecell>

remap_calib = cv2.remap(src = calib[side], map1 = map1, map2 = map2, interpolation = cv2.INTER_CUBIC, borderMode = cv2.BORDER_WRAP)

# <codecell>

showimage(remap_calib, size = 7)

# <codecell>

common_objectpoints = {0: None, 1: None}

commonpoints = {'left': {0: None, 1: None},
                'right': {0: None, 1: None}}

# <codecell>

plane = [0, 1][1]

cp_left = []
cp_right = []

for num, o in enumerate(objectpoints['left'][plane]):
    d_o = np.linalg.norm(o - objectpoints['right'][plane], axis = 1)
    
    if np.min(d_o) == 0:
        ind_o = np.argmin(d_o)
        
        cp_left.append(num)
        cp_right.append(ind_o)
        
common_objectpoints[plane] = objectpoints['left'][plane][cp_left]

commonpoints['left'][plane] = imagepoints['left'][plane][cp_left]
commonpoints['right'][plane] = imagepoints['right'][plane][cp_right]

# <codecell>

# sort the points in terms of increasing y values
sorty = np.argsort(common_objectpoints[plane][:, 1])

# <codecell>

common_objectpoints[plane] = common_objectpoints[plane][sorty]

commonpoints['left'][plane] = commonpoints['left'][plane][sorty]
commonpoints['right'][plane] = commonpoints['right'][plane][sorty]

# <codecell>

all_commonpoints = dict(left = np.vstack((commonpoints['left'][0], commonpoints['left'][1])),
                        right = np.vstack((commonpoints['right'][0], commonpoints['right'][1])))

# <codecell>

plane = [0, 1][1]

F, mask = cv2.findFundamentalMat(points1 = commonpoints['left'][0], # all_commonpoints['left'], 
                                 points2 = commonpoints['right'][0], # all_commonpoints['right'], 
                                 method = cv2.FM_LMEDS, # cv2.FM_RANSAC, 
                                 param1 = 0.5, param2 = 0.99)

# <codecell>

mask = mask.squeeze() == 1

# <codecell>

side = 'right'
plt.imshow(calib[side], cmap = mpl.cm.gray)

plt.plot(all_commonpoints[side][:, 0], all_commonpoints[side][:, 1], 'ro', mfc = 'None', mec = 'red', ms = 10)

plt.plot(all_commonpoints[side][mask][:, 0], all_commonpoints[side][mask][:, 1], 'go', mec = 'green', mfc = 'None', ms = 10)

plt.xlim(xmin = 0)
plt.ylim(ymin = 0)

plt.show()

# <codecell>

F

# <codecell>


epipolar_lines = cv2.computeCorrespondEpilines(points = commonpoints['left'][0], # all_commonpoints['left'], 
                                               whichImage = 1, F = F).squeeze()

a, b, c = epipolar_lines[0, :]
m1 = -a / b
t1 = -c / b

a, b, c = epipolar_lines[-1, :]
m2 = -a / b
t2 = -c / b

x_intersect = (t2 - t1) / (m1 - m2)
y_intersect = m1 * x_intersect + t1

# <codecell>

print x_intersect
print y_intersect

# <codecell>

side = 'right'

imcalib = calib[side]
ymax, xmax = imcalib.shape

fig, ax = plt.subplots(figsize = (15, 15))

# plt.axis('equal')

ax.imshow(imcalib, cmap = mpl.cm.gray)

x0 = min(0, x_intersect)
x1 = max(xmax, x_intersect)

y0 = min(0, y_intersect)
y1 = max(ymax, y_intersect)

for num, (a, b, c) in enumerate(epipolar_lines[[0, -1], :]):
    m = -a / b # slope
    t = -c / b # offset
    ax.plot([x0, x1], [m * x0 + t, m * x1 + t])
    
    ax.plot(all_commonpoints[side][[0, -1], :][num, 0], all_commonpoints[side][[0, -1], :][num, 1], 
            'ro', ms = 10, mec = 'red', mfc = 'None')

plt.xlim(xmin = x0, xmax = x1)
plt.ylim(ymin = y0, ymax = y1)

plt.show()

# <codecell>

cv2.getPerspectiveTransform()

# <codecell>

cv2.initCameraMatrix2D()

# <codecell>

# cv2.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize[, cameraMatrix1[, distCoeffs1[,
# cameraMatrix2[, distCoeffs2[, R[, T[, E[, F[, criteria[, flags]]]]]]]]]]) 
# -> retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F

allflags = [# cv2.CALIB_FIX_INTRINSIC, # fix cameraMatrix? and distCoeffs?, only calculate R, T, E, F matrices
            cv2.CALIB_USE_INTRINSIC_GUESS, # provide initial values
            # cv2.CALIB_FIX_PRINCIPAL_POINT,
            # cv2.CALIB_FIX_FOCAL_LENGTH,
            # cv2.CALIB_FIX_ASPECT_RATIO,
            # cv2.CALIB_SAME_FOCAL_LENGTH,
            # cv2.CALIB_FIX_K2,                # corresponding radial distortion coefficient is not changed
            # cv2.CALIB_FIX_K3,                # should be fixed unless fish-eye lens used; usually very small
            # cv2.CALIB_RATIONAL_MODEL,      # Coefficients k4, k5, and k6 are enabled
            # cv2.CALIB_ZERO_TANGENT_DIST,     # Tangential distortion coefficients (p1, p2) are set to zeros and stay zero
            ]

cameramatrix1 = all_cameramatrix['left'].copy()
cameramatrix2 = all_cameramatrix['right'].copy()
distcoeffs1 = all_dist_coefs['left'].copy()
distcoeffs2 = all_dist_coefs['right'].copy()

R = np.ones((3, 3))
T = np.ones((3, 1))
E = np.ones((3, 3))
F = np.ones((3, 3))

# <codecell>

rms = []
for k in range(100):
    out = cv2.stereoCalibrate(objectPoints = [common_objectpoints[0], common_objectpoints[1]], 
                    imagePoints1 = [commonpoints['left'][0], commonpoints['left'][1]],
                    imagePoints2 = [commonpoints['right'][0], commonpoints['right'][1]], 
                    imageSize = (image_width, image_height),
                    cameraMatrix1 = cameramatrix1,
                    distCoeffs1 = distcoeffs1,
                    cameraMatrix2 = cameramatrix2,
                    distCoeffs2 = distcoeffs2,
                    R = R, T = T, E = E, F = F,
                    flags = sum(allflags))
    rms.append(out[0])

# <codecell>

plt.plot((rms))
# plt.ylim(ymax = 4)
plt.show()

# <codecell>

cv2.stereoRectifyUncalibrated()

# <codecell>

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1 = cameramatrix1, 
                  distCoeffs1 = distcoeffs1, 
                  cameraMatrix2 = cameramatrix2, 
                  distCoeffs2 = distcoeffs2, 
                  imageSize = (image_width, image_height), 
                  R = R, T = T,
                  flags = 0 * cv2.CALIB_ZERO_DISPARITY,
                  alpha = 1,
                  newImageSize = (1 * image_width, 1 * image_height)
                  )

# <codecell>

map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix = cameramatrix1, 
                                         distCoeffs = distcoeffs1, 
                                         R = R1, 
                                         newCameraMatrix = P1,
                                         size = (1 * image_width, 1 * image_height), m1type = cv2.CV_32FC1)
side = ['left', 'right'][0]

# <codecell>

remap_calib = cv2.remap(src = calib[side],
                        map1 = map1, map2 = map2, 
                        interpolation = cv2.INTER_CUBIC, 
                        borderMode = cv2.BORDER_WRAP)

# <codecell>

showimage(remap_calib, size = 8)

# <codecell>

# computing stereo correspondence using the block matching algorithm
cv2.StereoBM()

# <codecell>

# computing stereo correspondence using the semi-global block matching algorithm
cv2.StereoSGBM()

# see example at /extra/usr/share/doc/packages/opencv-doc/examples/python2/stereo_match.py

# <rawcell>

# The class implements the modified S. G. Kosov algorithm [Publication] that differs from the original one as follows:
# 
# The automatic initialization of method’s parameters is added.
# The method of Smart Iteration Distribution (SID) is implemented.
# The support of Multi-Level Adaptation Technique (MLAT) is not included.
# The method of dynamic adaptation of method’s parameters is not included.

# <codecell>

# computing stereo correspondence using the variational matching algorithm
cv2.StereoVar()

# <codecell>

cv2.triangulatePoints()

# <codecell>

hemicine = Cine(initialdir = os.path.join(hemi_dir, 'left PCA/range finding'), 
                cinefilename = "left PCA range finding_023_Wed Oct 17 2012 14 21 49.538 317.001.cine", debug = True) 

# <codecell>

hemiframe = landscape( hemicine.getframe(0) )[0]

hemiframe = rotate( hemiframe, angle_deg = rotate_frame_calib )

# <codecell>

outfilename = "dataexchange.npz"

savez_dict = dict(calib = dict(left = hemiframe[:, :x_split], 
                               right = hemiframe[:, x_split:]), 
                  kp_coord = kp_coord) # kp_x = kp_x, kp_y = kp_y)

np.savez_compressed(outfilename, **savez_dict)

# <codecell>

showimage(hemiframe[:, :x_split], size = 9)

# <codecell>

Nwindow = 10

filt_hemi = simple_blur(hemiframe, # [:, :x_split], 
                        blocksize = Nwindow)

hemi = hemiframe # [:, :x_split]
clean_hemi = hemi - filt_hemi

# <codecell>

showimage(clean_hemi, size = 15)

# <codecell>

clean_hemi.dtype

# <codecell>

hist_hemi = equalize_hist(hemiframe) #[:, :x_split])
showimage(hist_hemi, size = 9)

# <codecell>

morph_hemi = erode(hist_hemi, blocksize = 3, Niterations = 3)
showimage(morph_hemi, size = 9)

# <codecell>

morph_hemi = erode(clean_hemi, blocksize = 3, Niterations = 3)
showimage(morph_hemi, size = 9)

# <codecell>

bin_hemi = adaptive_threshold(morph_hemi, blocksize = 40, offset = 30)
showimage(bin_hemi, size = 9)

# <codecell>

blur_hemi = gaussian_blur(hemiframe, blocksize = 50, sigma = 3)
showimage(blur_hemi, size = 9)

# <codecell>

hemi_keypoints = blobdetect(morph_hemi) # detector.detect(convto8(morph_hemi)) # clean_hemi))

# <codecell>

hemi_kp = np.array([k.pt for k in hemi_keypoints])
hemi_kp.shape

# <codecell>

plt.figure(figsize = (16, 16))
plt.imshow(hemiframe, cmap = mpl.cm.gray)
plt.plot(hemi_kp[:, 0], hemi_kp[:, 1], 'ro', ms = 10, mfc = 'None', mec = 'red')
plt.xlim(xmin = 50, right = 550)
plt.ylim(ymin = 120, top = 350)
plt.show()

# <codecell>

%%file maskselection.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# see lasso_demo.py
# /extra/usr/share/doc/packages/python-matplotlib/examples/event_handling/lasso_demo.py
# /extra/usr/share/doc/packages/python-matplotlib/examples/widgets/lasso_selector_demo.py
# http://matplotlib.org/examples/widgets/lasso_selector_demo.html
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

outfilename = "dataexchange.npz"

data = np.load(outfilename)

calib = data['calib'].tolist()
kp_coord = data['kp_coord'].tolist()

def onselect(vertices):
    path = Path( vertices, closed = True)
    
    # fraction of a pixel difference below which vertices will be simplified out
    path.simplify_threshold = 1.0
    path.should_simplify = True
    
    # path.contains_points(points = )

    vertfilename = "maskselection.npz"
    
    savez_dict = dict(vertices = vertices)
    
    np.savez_compressed(vertfilename, **savez_dict)
   
for showside in ['left']: # , 'right']:
    kp_x = [kp[0] for kp in kp_coord[showside]]
    kp_y = [kp[1] for kp in kp_coord[showside]]
    
    fig, ax = plt.subplots(figsize = (12, 12))
    fig.suptitle('calibration: %s' % showside)
    fig.canvas.manager.set_window_title('Press any key to finish mask selection')
    
    ax.imshow( calib[showside], cmap = mpl.cm.gray, zorder = 1)
    
    # ax.plot(kp_x, kp_y, 'ro', mfc = 'None', mec = 'red', ms = 20, zorder = 10)
    
    # plt.grid(True)
    
    # plt.xlim(xmin = 0) #, xmax = 325)
    # plt.ylim(ymax = 0) #, ymax = 375)

    lasso = LassoSelector(ax = ax, onselect = onselect, useblit = True,
                          lineprops = dict(color = 'yellow', linewidth = 7, linestyle = 'dashed', marker = 'None'))
    
    print "Press any key to finish mask selection"
    while True:
        buttonpressed = plt.waitforbuttonpress(timeout = -1)
        if buttonpressed:
            break

lasso.disconnect_events()
fig.canvas.manager.set_window_title('Mask selection DONE')

plt.show()

# <codecell>

%%script python 
# --bg --err stderror_pipe --out stdout_pipe
import sys
import numpy as np

print sys.argv
print 'Python version: %s' % sys.version

import matplotlib as mpl
import matplotlib.pyplot as plt

print "mpl: ", mpl.get_backend()

import maskselection

# <codecell>

import matplotlib.path as mpath
from matplotlib.path import Path

vertfilename = "maskselection.npz"

data = np.load(vertfilename)

verts = data['vertices']

path = Path(verts, closed = True)
print len(path)

path.should_simplify = True
path.simplify_threshold = 10.0

# <codecell>

vertices = []
codes = []
for (vertex, code) in path.iter_segments(simplify = True):
    vertices.append(vertex.tolist())
    codes.append(code)

# <codecell>

cleanpath = Path(vertices, codes)
len(cleanpath)

# <codecell>

codes[-1] = Path.CLOSEPOLY

# <codecell>

bbox = cleanpath.get_extents()
print bbox

# <codecell>

bbox.bounds

# <codecell>

hemi_masked = path.contains_points( hemi_kp )
hemi_kp_masked = hemi_kp[ np.where(hemi_masked) ]

# <codecell>

fig, axs = plt.subplots(1, 2, figsize = (15, 15))

ax = axs[0]

p = mpl.patches.PathPatch(path, ls = 'solid', lw = 2, edgecolor = 'k', facecolor = 'None', zorder = 100, )
ax.add_patch(p)

cleanp = mpl.patches.PathPatch(cleanpath, ls = 'solid', lw = 2, edgecolor = 'y', facecolor = 'None', zorder = 100, )
ax.add_patch(cleanp)

showside = 'left'
ax.imshow(dict(left = hemiframe[:, :x_split], right = hemiframe[:, x_split:])[showside], 
          cmap = mpl.cm.gray, zorder = 1)

ax.plot(hemi_kp_masked[:, 0], hemi_kp_masked[:, 1], 'ro', ms = 10, mfc = 'None', mec = 'red')

ax.set_xlim(left = 50, right = 350)
ax.set_ylim(bottom = 100, top = 350)

axs[1].imshow(morph_hemi, cmap = mpl.cm.gray)
axs[1].plot(hemi_kp_masked[:, 0], hemi_kp_masked[:, 1], 'ro', ms = 10, mfc = 'None', mec = 'red')

axs[1].set_xlim(left = 50, right = 350)
axs[1].set_ylim(bottom = 100, top = 350)

plt.show()

# <codecell>

path.contains_points(points = np.array([[120, 120], [140, 140]]),  )

# <codecell>


