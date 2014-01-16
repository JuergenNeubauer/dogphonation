# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, sys

# <codecell>

sys.path.append('./tools/')

# <codecell>

from IPython.parallel import Client
from IPython.parallel import require

# <codecell>

client = Client()

# <codecell>

directview = client[:]

# <codecell>

print client.ids
Ncpus = len(client.ids)

# <codecell>

import readTDMS
import readTDMS as tdms

# <codecell>

rootdir = '/mnt/workspace/InVivoDog_2013_12_11/'

# <codecell>

!ls -alot $rootdir

# <codecell>

def test(rootdir = '/mnt/workspace'):
    import sys
    sys.path.append('/extra/public/python/tools/')
    import readTDMS as tdms
    return tdms.datatdms(rootdir = rootdir)

# <codecell>

r = directview.apply_sync(test, rootdir = rootdir)

# <codecell>

ls -alo $rootdir/data\ LabView

# <codecell>

tdmsfiles = tdms.datatdms(rootdir = rootdir)

# <codecell>

def process_tdmsfile(tdmsfile_info, rerun = False):
    import os, sys
    sys.path.append('/extra/public/python/tools/')
    import readTDMS as tdms
    
    casedir, casename = tdmsfile_info
    
    hdf5name = os.path.join(casedir, casename.replace('.tdms', '.hdf5'))
    
    if rerun == False:
        # skip already converted files
        if os.path.isfile(hdf5name):
            if os.path.getsize(hdf5name) > 0:
                return
        
    # print casedir + ": " + casename
    
    metadata, rawdata, datafilename = tdms.readfile(casedir, casename)
    
    expdata = tdms.tdmstodict(metadata, rawdata)
    
    del rawdata, metadata
    
    recordings = tdms.formatrecordings(expdata)
    
    del expdata
    
    tdmsfilename = datafilename
    hdf5filename = tdms.savehdf5(tdmsfilename, recordings)
    
    return casedir + ": " + casename

# <markdowncell>

# # serial run on only ONE cpu
# for tdmsfile_info in tdmsfiles:
#     process_tdmsfile(tdmsfile_info, rerun = False)

# <codecell>

rerun = [False] * len(tdmsfiles)

# <codecell>

directview.map(process_tdmsfile, tdmsfiles, rerun, block = True)

# <codecell>

len(_)

# <codecell>

import wavfileconversion as wavconversion

# <markdowncell>

# # serial processing on ONE cpu
# wavconversion.process(rootdir = rootdir)

# <codecell>

def process_wavconversion(tdmsfile_info):
    import os, sys
    sys.path.append('/extra/public/python/tools/')
    
    import wavfileconversion as wavconversion
    
    wavconversion.process(tdmsfile_info)

# <codecell>

directview.map(process_wavconversion, tdmsfiles, block = True)

# <codecell>

len(_)

# <codecell>


