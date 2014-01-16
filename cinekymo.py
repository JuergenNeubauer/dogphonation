# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os
import numpy as np

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')
sys.path.append('/extra/public/python/tools')

# <codecell>

import cine

# <codecell>

try:
    fullcinefilepath = sys.argv[1]
except:
    fullcinefilepath = None
else:
    if not os.path.isfile(fullcinefilepath):
        # for debugging and testing from IPython Notebook
        fullcinefilepath = ("/extra/InVivoDog/InVivoDog_2012_10_17/left PCA/range finding/" + 
                            "left PCA range finding_006_Wed Oct 17 2012 14 18 49.507 050.001.cine")

# <codecell>

if fullcinefilepath:
    initialdir = os.path.dirname(fullcinefilepath)
    cinefilename = os.path.basename(fullcinefilepath)
else:
    initialdir = None
    cinefilename = None
print initialdir
print cinefilename

# <codecell>

c = cine.Cine(initialdir = initialdir, 
              cinefilename = cinefilename,
              debug = True)

# <codecell>

try:
    kymolinenr = np.int(sys.argv[2])
except:
    kymolinenr = 225

# <codecell>

kymo = c.makekymo(cineframes = None, linenr = kymolinenr, orientation = 'horizontal')

# <codecell>

np.savez_compressed(file = '/tmp/cinekymo.npz', kymo = kymo)

# <codecell>

c.getallframes()

# <codecell>

try:
    debug = sys.argv[3]
except:
    debug = False

# <codecell>

if debug:
    import matplotlib.pyplot as plt
    
    plt.imshow(kymo, aspect = 'auto')
    plt.gray()

    plt.title('kymo linenr: %d' % kymolinenr)
    
    plt.show()

# <codecell>


