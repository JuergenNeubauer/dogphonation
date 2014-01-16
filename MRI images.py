# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os
import numpy as np

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib.pyplot as plt

# <codecell>

import nibabel as nib

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools/')

# <codecell>

%run "tools for hemilarynx analysis.py"

# <codecell>

MRI_dir = '/extra/InVivoDog/InVivoDog_2013_10_30/MRI'

# <codecell>

!ls -alot $MRI_dir/*.img

# <codecell>

MRI_data = 'larynx_T2w_RARE3.hdr'
# MRI_data = 'larynx_t2w_surfacecoil.hdr'
# MRI_data = 'larynx_T2w_inGaldin.hdr'
MRI_data = 'larynx_T2w.hdr'
MRI_data = 'doglarynx_T2w.hdr'

# <codecell>

img = nib.load(os.path.join(MRI_dir, MRI_data))

# <codecell>

img.shape

# <codecell>

data = img.get_data().squeeze()
data.shape

# <codecell>

data.max()

# <codecell>

hist_data, cdf = histeq(data[:, :, 15], number_bins = 2**20)

# <codecell>

plt.imshow(data[:, :, 15])

# <codecell>

plt.imshow(hist_data)
plt.gray()

# <codecell>

plt.gray()

# <codecell>

data.shape

# <codecell>

for k in range(data.shape[2]):
    plt.imshow(data[:, :, k])
    a = plt.gca()
    a.set_xticklabels([])
    a.set_yticklabels([])
    
    savename = os.path.splitext(MRI_data)[0]
    plt.savefig('%s_slice_%03d.png' % (savename, k))

# <codecell>


