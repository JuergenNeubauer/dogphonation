# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os, csv, glob
import numpy as np

# %pylab inline

import matplotlib as mpl
# use the backend for inline plots in the IPython Notebook
# so then don't need to use magic pylab with inline option
mpl.use('module://IPython.kernel.zmq.pylab.backend_inline')

import matplotlib.pyplot as plt

print "Matplotlib will use backend: ", mpl.get_backend()
print "Pyplot will use backend: ", plt.get_backend()

# <codecell>

from IPython.display import HTML
def webframe(urlstring, width = 1000, height = 500):
        return HTML("<iframe src=%s width=%d height=%d></iframe>" % (urlstring, width, height))

# <codecell>

webframe("http://jneubaue.bol.ucla.edu/publications/JAS03179.pdf", width = 1300)

# <markdowncell>

# analysis for JASA paper is in:
# 
# /extra/backup_lighthill/homelighthill/neubauer/documents/jasapaper/figures/matlab

# <codecell>

sys.path.append('./tools/')

# <codecell>

from pca import PCA

# <codecell>

help PCA

# <codecell>


