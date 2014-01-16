# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# OLD CODE
# =========

# <codecell>

hemi_dir = '/mnt/workspace/InVivoDog_2012_10_10/left TA/range_finding'

hemi_filename = 'left TA range_finding_071_Wed Oct 10 2012 15 02 57.492 437.001.cine'

hemi_dir = '/mnt/workspace/InVivoDog_2012_10_10/TA-trunkRLN/NoSLN/'
hemi_filenames = sorted(glob.glob(os.path.join(hemi_dir, '*.cine')))
print "number of cine files in this directory: ", len(hemi_filenames)

stimindex = 5 * 5 + 1

hemi_filename = os.path.basename(hemi_filenames[stimindex - 1])

if False:
    hemi_dir = '/mnt/workspace/InVivoDog_2012_10_10/calibration/'
    hemi_filename = 'calibration_001_Wed Oct 10 2012 14 40 21.189 567.001_Cam_11440_Cine2.cine'

    hemi_filename = None

# <codecell>

hemi_dir = '/mnt/workspace/InVivoDog_2012_10_17/'
hemi_filename = None

# <codecell>

exp_cine_dirs = glob.glob(os.path.join(hemi_dir, 'left*'))

exp_cine_dirs = [hemi_dir]
print "experimental cine dirs: ", exp_cine_dirs

for expdir in exp_cine_dirs:
    for dirpath, dirnames, filenames in os.walk(expdir):
        print "dirpath: ", dirpath
        for filename in filenames:
            if not filename.endswith('.cine'):
                continue
            c = Cine(initialdir = dirpath, cinefilename = filename, debug = True)
            
            fall = c.getallframes()

            raw_frames_dir = c.cinefilename.replace('.cine', '.frames_raw')
            os.makedirs(raw_frames_dir)
            
            for num, frame in enumerate(fall):
                raw_cine_name = c.cinefilename.replace('.cine', '.frame%04d.raww' % num)
                raw_cine_filepath = os.path.join(raw_frames_dir, os.path.basename(raw_cine_name))
                frame.tofile(raw_cine_filepath)

print "exported all frames from cine to binary file (raww: uint16)"
print "pixel geometry: ", frame.shape

# <codecell>

%connect_info

# <codecell>

c = Cine(initialdir = hemi_dir, cinefilename = hemi_filename, debug = True)
print "read cine file: ", c.cinefilename

# <codecell>

print "Npixel: ", c.Npixel
print "frame rate: ", c.framerate
print "imagecount: ", c.imagecount
print "imagesize: ", c.imagesize
print "bitmapheader: ", c.bitmapheader
print "frame height: ", c.height
print "frame width: ", c.width

# <codecell>

import time
f0 = dict()
c.debug = False

t0 = time.time()
for num, framenum in enumerate(range(0, 300, 3)):
    f0[num] = c.getframe(framenum)[0]

dt = time.time() - t0
f0nbytes = sum(np.array([item.nbytes for item in f0.values()]))

print "disk reading time: %.3f milliseconds" % (dt * 1000.0)
print "disk reading rate: %.1f MB/s" % (f0nbytes / dt / 2.**20)

print "shape: ", f0[0].shape

# <codecell>

imshow(f0[0], aspect = 'equal', origin = 'upper')
grid(False)

# <codecell>

imshow(np.fliplr(f0[0].T), aspect = 'equal', origin = 'upper')
grid(False)

# <codecell>

imshow(np.fliplr(f0[0][ 200:, :].T), aspect = 'equal', origin = 'upper')
grid(False)

# <codecell>

c.debug = True
fall = c.getallframes()

# <codecell>

raw_frames_dir = c.cinefilename.replace('.cine', '.frames_raw')
os.makedirs(raw_frames_dir)

# <codecell>

for num, frame in enumerate(fall):
    raw_cine_name = c.cinefilename.replace('.cine', '.frame%04d.raww' % num)
    raw_cine_filepath = os.path.join(raw_frames_dir, os.path.basename(raw_cine_name))
    frame.tofile(raw_cine_filepath)

print "exported all frames from cine to binary file (raww: uint16)"
print "pixel geometry: ", frame.shape

# <codecell>

imshow(frame, origin = 'upper', aspect = 'equal')

# <codecell>

thumb_every_N_millisec = 5
thumbstrip = fall[::thumb_every_N_millisec * 3, 400::6, ::6]
print "", fall.shape
print "", thumbstrip.shape
Nt, ht, wt = thumbstrip.shape

# <codecell>

Nthumbs = 15
print "thumbstrip shows %d millisec" % (Nthumbs * thumb_every_N_millisec)
thumbs = np.empty((wt, Nthumbs * ht), dtype = thumbstrip.dtype)
for num, frame in enumerate(thumbstrip[:Nthumbs]):
    thumbs[:, num*ht:(num+1)*ht] = np.fliplr(frame.T)

# <codecell>

imshow(thumbs, 
       aspect = 'equal', 
       origin = 'upper')
grid(False)

# <codecell>

kymo = fall[:, 525, :]

# <codecell>

imshow(kymo[:300].T, origin = 'upper')
xlabel('time [frames]')

# <codecell>

import Image

# <codecell>

Image.fromarray(f0[0].astype(np.uint8))

# <codecell>

imshow(f0[0].astype(np.uint8))

# <codecell>

kymo = c.makekymo(None, linenr = 525)

# <codecell>

imshow(kymo, origin = 'lower', aspect = 'auto')

# <codecell>


