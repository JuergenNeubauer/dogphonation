# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os
import numpy as np

sys.path.append("/extra/InVivoDog/python/cine/tools")

from dogdata import DogData

datadir = "/extra/InVivoDog/InVivoDog_2012_03_21/data LabView/SLN_trunkRLN/"

datafiles = {"TA 0": "SLN_trunkRLN_NoTA Wed Mar 21 2012 14 46 34.hdf5",
             "TA 1": "SLN_trunkRLN_ThresholdTA_condition01 Wed Mar 21 2012 14 55 08.hdf5",
             "TA 2": "SLN_trunkRLN_TA_condition02 Wed Mar 21 2012 15 01 30.hdf5",
             "TA 3": "SLN_trunkRLN_TA_condition03 Wed Mar 21 2012 15 09 20.hdf5",
             "TA 4": "SLN_trunkRLN_MaxTA_condition04 Wed Mar 21 2012 15 17 43.hdf5"}

TAconditions = sorted(datafiles.keys())

# <codecell>

import csv, os

csvfile = csv.reader(open("/extra/InVivoDog/Dinesh/03_21_2012_SLNvsTrunk.csv", 'r'), 
                     dialect = 'excel')

# skip first three lines
for k in range(3):
    csvfile.next()

F0 = [] # onset frequency in Hz
onset = [] # onset time in milliseconds

for row in csvfile:
    F0block = row[1:6]
    onsetblock = row[8:13]

    F0block_clean = []
    onsetblock_clean = []

    for item in F0block:
        if item not in ['NP', '0', '#VALUE!']:
            F0block_clean.append(float(item))
        else:
            F0block_clean.append(np.nan)

    for item in onsetblock:
        if item not in ['NP', '0', '#VALUE!']:
            onsetblock_clean.append(float(item))
        else:
            onsetblock_clean.append(np.nan)

    F0.append(F0block_clean)
    onset.append(onsetblock_clean)

del csvfile

# <codecell>

F0 = np.array(F0)
onset = np.array(onset)

ps_onset = np.ones(onset.shape) * np.nan
Q_onset = np.ones(onset.shape) * np.nan

# Bernoulli equivalent area A proportional to Q / sqrt(p)
A_onset = np.ones(onset.shape) * np.nan

# <codecell>

for TAnum, TAcond in enumerate(TAconditions):
    dogdata = DogData(datadir = datadir, datafile = datafiles[TAcond])
    print "working on: ", dogdata.datafilename
    dogdata.get_all_data()
    
    time = dogdata.time_psQ * 1000.0 # onset time is given in milliseconds
    
    for stimind, datarow in enumerate(dogdata.allps):
        if not np.isnan(onset[stimind, TAnum]):
            ps_onset[stimind, TAnum] = np.interp(onset[stimind, TAnum], time, datarow)

    for stimind, datarow in enumerate(dogdata.allQ):
        if not np.isnan(onset[stimind, TAnum]):
            Q_onset[stimind, TAnum] = np.interp(onset[stimind, TAnum], time, datarow)

A_onset = Q_onset / np.sqrt(ps_onset)

# <codecell>

def distances(clickdata):
    # baseline vectors between anterior landmark and VP on left and right sides
    vlr_clickdata = clickdata[1:] - clickdata[0]
    # vector between left and right VPs
    dx_clickdata = clickdata[1] - clickdata[2]
    
    # baseline lengths
    l_clickdata = np.hypot(vlr_clickdata[:, 0], vlr_clickdata[:, 1])
    d_clickdata = np.hypot(dx_clickdata[0], dx_clickdata[1])
    
    return l_clickdata, d_clickdata

# <codecell>

# landmarks at rest and onset clicked by Eli
# if onset didn't occur, used the last frame in recording
# see strainanalysis.py
landmarkdir = "/extra/InVivoDog/python/cine/results_save"

# <codecell>

import glob, os

clickfiles = sorted(glob.glob(os.path.join(landmarkdir, '*.npz')))

# <codecell>

strains = {}

for TAnum, TAcond in enumerate(TAconditions):
    strains[TAcond] = []

    TAcond_name = datafiles[TAcond].split()[0]

    TAcond_landmarkfiles = sorted(
        [item for item in clickfiles if os.path.basename(item).startswith(TAcond_name)])

    for clickfile in TAcond_landmarkfiles:
        data = np.load(clickfile)

        if data['TAconditionindex'] != TAnum:
            print "Error: wrong TA condition index"
            continue

        baseline = data['baseline_pos']
        if len(baseline) < 3:
            print "ERROR: baseline: #clicks < 3: ", clickfile
            strains.append([np.nan, np.nan, np.nan])
            continue
        onset = data['onset_pos']
        if len(onset) < 3:
            print "ERROR: onset: #clicks < 3: ", clickfile
            strains.append([np.nan, np.nan, np.nan])
            continue

        l_baseline, d_baseline = distances(baseline)
        l_onset, d_onset = distances(onset)

        leftstrain, rightstrain = (l_onset - l_baseline) / l_baseline * 100.0
        # dVPrel = (d_onset - d_baseline) / d_baseline * 100.0
        dVPrel = d_onset / d_baseline * 100.0

        # FIX outlier by hand: TA condition 2, clickfile 11 (counting starts from 0)
        if TAnum == 2:
            if clickfile == TAcond_landmarkfiles[11]:
                leftstrain = strains[TAcond][9][0] * 1.05
                rightstrain = strains[TAcond][9][1] * 1.07
                dVPrel = strains[TAcond][9][2] * 0.94

        strains[TAcond].append([leftstrain, rightstrain, dVPrel])

# <codecell>

F0_plot = {}
ps_plot = {}
Q_plot = {}
A_plot = {}

lstrain_plot = {}
rstrain_plot = {}
dVP_plot = {}

Nlevels = dogdata.Nlevels
# dogdata.nervenamelist
# ['left SLN', 'right SLN', 'left RLN', 'right RLN', 'left TA', 'right TA']
# we need SLN versus RLN trunk
rellevels = dogdata.a_rellevels[:, [0, 2]]

for TAnum, TAcond in enumerate(TAconditions):
    F0_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    ps_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    Q_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    A_plot[TAcond] = np.zeros((Nlevels, Nlevels))

    lstrain_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    rstrain_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    dVP_plot[TAcond] = np.zeros((Nlevels, Nlevels))

    for stimind, (SLNlevel, RLNlevel) in enumerate(rellevels):
        # SLN: row index, i.e. y-axis in plot
        # RLN: column index, i.e. x-axis in plot
        F0_plot[TAcond][SLNlevel, RLNlevel] = F0[stimind, TAnum]
        ps_plot[TAcond][SLNlevel, RLNlevel] = ps_onset[stimind, TAnum]
        Q_plot[TAcond][SLNlevel, RLNlevel] = Q_onset[stimind, TAnum]
        A_plot[TAcond][SLNlevel, RLNlevel] = A_onset[stimind, TAnum]

        lstrain_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][0]
        rstrain_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][1]
        dVP_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][2]

        if np.isnan(F0_plot[TAcond][SLNlevel, RLNlevel]):
            lstrain_plot[TAcond][SLNlevel, RLNlevel] = np.nan
            rstrain_plot[TAcond][SLNlevel, RLNlevel] = np.nan
            dVP_plot[TAcond][SLNlevel, RLNlevel] = np.nan

# <codecell>

all_F0 = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_ps = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_Q = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_A = np.zeros( (Nlevels, len(TAconditions), Nlevels) )

all_lstrain = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_rstrain = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_dVP = np.zeros( (Nlevels, len(TAconditions), Nlevels) )

for TAnum, TAcond in enumerate(TAconditions):
    all_F0[:, TAnum, :] = F0_plot[TAcond] # SLNlevel, TAlevel, trunkRLNlevel
    all_ps[:, TAnum, :] = ps_plot[TAcond]
    all_Q[:, TAnum, :] = Q_plot[TAcond]
    all_A[:, TAnum, :] = A_plot[TAcond]

    all_lstrain[:, TAnum, :] = lstrain_plot[TAcond]
    all_rstrain[:, TAnum, :] = rstrain_plot[TAcond]
    all_dVP[:, TAnum, :] = dVP_plot[TAcond]

# <codecell>

import matplotlib.pyplot as plt

plt.close('all')

# <codecell>

figurenametemplate = 'SLN-trunkRLN'

label_trunkRLN = 'LCA/IA level' # 'trunk RLN level'
label_SLN = 'CT level' # 'SLN level'
label_TA = 'TA level'

# <codecell>

def plotdata(TAcond, figlabel = 'onset', ):
    figname = 'onset frequency: %s' % TAcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(F0_plot[TAcond])
    ax = plt.gca()
    
    # axim.set_clim(100, 720)
    axim.set_clim(np.nanmin(all_F0), np.nanmax(all_F0))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 7.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_trunkRLN)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset frequency [Hz]')

    figurename = '%s.onsetF0.%s.pdf' % (figurenametemplate, TAcond.replace(' ', ''))  
    plt.savefig(figurename, format = 'pdf', orientation = 'landscape',
                bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>


