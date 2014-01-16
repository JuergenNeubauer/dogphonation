# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from IPython.display import SVG

# <codecell>

import sys
sys.path.append('/extra/InVivoDog/python/cine/tools')

# <codecell>

sys.modules

# <codecell>

import subplotgrid
reload(subplotgrid)
from subplotgrid import Specgrid

# <codecell>

del specgrid

# <codecell>

specgrid = Specgrid(nrows = 8, ncols = 8, figsize = (18, 12))

# <codecell>

specgrid.fig.set_size_inches(24, 18, forward = True) # old values: 24, 12

# <codecell>

specgrid.test()

# <codecell>

specgrid.savefig(savename = 'test.pdf', backend = 'pdf')

# <codecell>

import dogdata
reload(dogdata)

# <codecell>

d = dogdata.DogData(datadir = '/extra/InVivoDog', datafile = 'trunkRLN_TA_NoSLN Wed Mar 21 2012 17 18 17.hdf5')

# <codecell>

d.Nlevels

# <codecell>

d.Nnerves

# <codecell>

d.Nrecnums

# <codecell>

d.nerveorder

# <codecell>

d.nervenamelist

# <codecell>

d.a_rellevels

# <codecell>

grid_xaxis = dict(label = 'RLN', level = 'leftRLN')
grid_yaxis = dict(label = 'TA', level = 'leftTA')

d.show_spectrograms(signal = 'psub', nerve_xaxis = grid_xaxis, nerve_yaxis = grid_yaxis)

# <codecell>

d.datafilename

# <codecell>

d.savefigure()

