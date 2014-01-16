# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, sys, glob
import numpy as np

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')

# <codecell>

%matplotlib inline

%config InlineBackend.close_figures = False

# <codecell>

import matplotlib as mpl
import matplotlib.pyplot as plt

print mpl.get_backend()
print plt.get_backend()

mpl.interactive(False)
plt.interactive(False)

print mpl.is_interactive()
print plt.isinteractive()

# <codecell>

import dogdata

# <codecell>

datadir = "/extra/InVivoDog/InVivoDog_2012_03_21/data LabView/SLN_trunkRLN/"

datafiles = {"TA 0": "SLN_trunkRLN_NoTA Wed Mar 21 2012 14 46 34.hdf5",
             "TA 1": "SLN_trunkRLN_ThresholdTA_condition01 Wed Mar 21 2012 14 55 08.hdf5",
             "TA 2": "SLN_trunkRLN_TA_condition02 Wed Mar 21 2012 15 01 30.hdf5",
             "TA 3": "SLN_trunkRLN_TA_condition03 Wed Mar 21 2012 15 09 20.hdf5",
             "TA 4": "SLN_trunkRLN_MaxTA_condition04 Wed Mar 21 2012 15 17 43.hdf5"}

TAconditions = sorted(datafiles.keys())

# <codecell>

d = dogdata.DogData(datadir = datadir, datafile = datafiles['TA 2'])

# <codecell>

[ [num] + l.tolist() for num, l in enumerate(d.a_rellevels)]

# <codecell>

d.get_all_data()

# <codecell>

fig1, ax1 = plt.subplots(figsize = (6, 6))
fig1.clf()

ax1.plot(d.allps[0, :], d.allQ[0, :], 'k')
ax1.plot(d.allps[15, :], d.allQ[15, :], 'r')

ax1.set_xlabel('ps')
ax1.set_ylabel('Q')

fig1

# <codecell>

from matplotlib.backends.backend_agg import FigureCanvasAgg as figurecanvas
canvas = figurecanvas(fig1)

# <codecell>

plt.close('all')
plt.clf()
plt.plot(d.allQ[25, :])
plt.show()

# <codecell>

import IPython
from IPython import display

def audio(url):
  return display.HTML("<audio controls><source src='{}'></audio>".format(url))
IPython.display.Audio = audio

# <codecell>

audio('http://people.sc.fsu.edu/~jburkardt/data/wav/thermo.wav')

# <codecell>

psub = d.allpsub[15, :]

# <codecell>

plt.close('all')
plt.plot(d.allpsub[15, 8000:20000], 'k-')
plt.show()

# <codecell>

t0 = 8000 + 10000
W = int( 25e-3 * 50000 ) # 5000
corr = np.correlate(psub[t0:t0 + 2 * W - 1], psub[t0:t0 + W], mode = 'valid')

# <codecell>

print W
print len(corr)

# <codecell>

plt.close('all')
plt.plot(corr, 'k')
plt.show()

# <codecell>

diff = np.zeros(W)

for tau in range(W):
    diff[tau] = np.sum( (psub[t0:t0 + W] - psub[t0 + tau:t0 + W + tau])**2 )

# <codecell>

plt.close('all')

plt.plot(diff, 'k')

plt.show()

# <codecell>


