# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
sys.path.append('/extra/InVivoDog/python/cine/tools')

# <codecell>

import dogdata
reload(dogdata)

# <codecell>

datadir = "/extra/InVivoDog/InVivoDog_2013_10_30/data LabView/"

# <codecell>

ls -alot "$datadir"

# <codecell>

import glob, os

# <codecell>

# need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
        
# for 2013_10_23:
Voffset_ps = -38.0e-3 # depends on the date
isoampgain_ps = 2.0 # I measured it more accurately, it is slightly more than 2

Voffset_Q = 33.0e-3 # depends on the date
isoampgain_Q = 2.0

# for 2013_10_30
Voffset_ps = 8.8e-3

Voffset_Q = 3.5e-3

for root, dirs, files in os.walk(top = datadir):
    if not files:
        continue
    if root.find('range') > 0: # exclude range finding files
        continue
    for hdf5file in files:
        if not hdf5file.endswith('.hdf5'):
            continue
        print hdf5file
                    
        d = dogdata.DogData(datadir = root, datafile = hdf5file)
        
        d.get_all_data()
        
        # need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
        Vconv = d.convEMG # EMG conversion is just conversion from numbers to Volts
        
        d.allps = isoampgain_ps * d.allps - Voffset_ps / Vconv * d.convps
        
        d.allQ = isoampgain_Q * d.allQ - Voffset_Q / Vconv * d.convQ
        
        d.minps = d.allps.min()
        d.maxps = d.allps.max()
        d.minQ = d.allQ.min()
        d.maxQ = d.allQ.max()

        if hdf5file.startswith('SLN versus RLN'): # No implant'):
            grid_xaxis = dict(label = 'RLN', level = 'rightRLN')
            grid_yaxis = dict(label = 'SLN', level = 'rightSLN')
            
        if hdf5file.startswith('SLN versus right RLN'): # ('SLN versus RLN, no left RLN'):
            grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
            grid_yaxis = dict(label = 'SLN', level = 'rightSLN')
            
        if hdf5file.startswith('right SLN versus right RLN'):
            grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
            grid_yaxis = dict(label = 'right SLN', level = 'rightSLN')
        
        for signal in ['psub', 'pout']:
            d.show_spectrograms(signal = signal, 
                                nerve_xaxis = grid_xaxis, nerve_yaxis = grid_yaxis,
                                figsize = (2*24/3, 2*18/3))
            
            d.show_data(nerve_xaxis = grid_xaxis, nerve_yaxis = grid_yaxis)
            
            d.savefigure(format = 'png')
            
            del d.allspecs
            
        del d

# <codecell>

specgramfiles = []

for root, dirs, files in os.walk(top = datadir):
    if not files:
        continue
    if root.find('range') > 0: # exclude range finding files
        continue
    for pngfile in files:
        if not pngfile.endswith('.png'):
            continue
        print pngfile
        specgramfiles.append(os.path.join(root, pngfile))

# <codecell>

checkonsetfiles = []

for root, dirs, files in os.walk(top = '/extra/InVivoDog/InVivoDog_2013_10_23/data LabView/'):
    if not files:
        continue
    if root.find('range') > 0: # exclude range finding files
        continue
    for pngfile in files:
        if not pngfile.endswith('.png'):
            continue
        if pngfile.find('Check_Onset') < 0:
            continue
        print pngfile
        checkonsetfiles.append(os.path.join(root, pngfile))

# <codecell>

import shlex, subprocess

# <codecell>

psubfiles = [item for item in specgramfiles if item.find('.psub.') > 0]
poutfiles = [item for item in specgramfiles if item.find('.pout.') > 0]

# <codecell>

string_psubfiles = " ".join(['"%s"' % item for item in psubfiles])
string_poutfiles = " ".join(['"%s"' % item for item in poutfiles])

# <codecell>

cmd = 'viewnior %s' % string_psubfiles

subprocess.check_output(shlex.split(cmd), stderr = subprocess.STDOUT)

# <codecell>

for item in poutfiles + psubfiles: # checkonsetfiles:
    cmd = 'ln -s "%s" ./Implant_2013_10_30/specgrams/' % item
    subprocess.check_output(shlex.split(cmd), stderr = subprocess.STDOUT)

# <markdowncell>

# cmd = 'zip implant_specgrams.zip %s' % all_psubfiles

# <codecell>


