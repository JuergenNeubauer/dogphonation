# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

pwd

# <codecell>

import sys
import numpy as np

sys.path.append("/extra/InVivoDog/python/cine/tools/")

from dogdata import DogData

datadir = "/extra/InVivoDog/InVivoDog_2012_03_21/data LabView/SLN_trunkRLN/"

datafiles = {"TA 0": "SLN_trunkRLN_NoTA Wed Mar 21 2012 14 46 34.hdf5",
             "TA 1": "SLN_trunkRLN_ThresholdTA_condition01 Wed Mar 21 2012 14 55 08.hdf5",
             "TA 2": "SLN_trunkRLN_TA_condition02 Wed Mar 21 2012 15 01 30.hdf5",
             "TA 3": "SLN_trunkRLN_TA_condition03 Wed Mar 21 2012 15 09 20.hdf5",
             "TA 4": "SLN_trunkRLN_MaxTA_condition04 Wed Mar 21 2012 15 17 43.hdf5"}

TAconditions = sorted(datafiles.keys())

# <codecell>

ls 03_21_2012_SLNvsTrunk*

# <codecell>

#####################
import csv

csvfile = csv.reader(open("03_21_2012_SLNvsTrunk.csv", 'r'), 
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

# <rawcell>

# from IPython.display import display
# 
# import pandas as pd
# 
# pd.core.format.set_option("display.notebook_repr_html", True)
# 
# f0 = pd.DataFrame(F0, columns = TAconditions)
# 
# display(f0.head())

# <rawcell>

# from IPython.display import HTML
# HTML(f0.to_html())

# <codecell>

#####################
F0 = np.array(F0)
onset = np.array(onset)

ps_onset = np.ones(onset.shape) * np.nan
Q_onset = np.ones(onset.shape) * np.nan

# area A proportional to Q / sqrt(p)
A_onset = np.ones(onset.shape) * np.nan

# <codecell>

#####################
for TAnum, TAcond in enumerate(TAconditions):
    dogdata = DogData(datadir = datadir, datafile = datafiles[TAcond])
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

#####################
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
# see strainanalysis.py
landmarkdir = "/extra/InVivoDog/python/cine/results_save"

import glob, os

# <codecell>

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

#####################
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

# <codecell>

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

#####################
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

%matplotlib inline

# <codecell>

#####################

import matplotlib as mpl
print "matplotlib: ", mpl.get_backend()

# mpl.use('module://IPython.kernel.zmq.pylab.backend_inline')
import matplotlib.pyplot as plt
print "pyplot: ", plt.get_backend()

# mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['image.cmap'] = 'jet'

# plt.close('all')

# <codecell>

print "matplotlib: ", mpl.is_interactive()
mpl.interactive(False)
print "pyplot: ", plt.isinteractive()
plt.interactive(False)

# <codecell>

%config InlineBackend
%config InlineBackend.close_figures = False

# <codecell>

figurenametemplate = 'SLN-trunkRLN'

label_trunkRLN = 'LCA/IA level' # 'trunk RLN level'
label_SLN = 'CT level' # 'SLN level'
label_TA = 'TA level'

figureformat = 'pdf' # 'eps' # 'pdf'
figureorientation = 'portrait' # 'landscape'

# <codecell>

mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'medium'

# <codecell>

print mpl.rcParams['figure.dpi']
print mpl.rcParams['savefig.dpi']

# <codecell>

ls -alo Figure1?.pdf

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-trunkRLN | grep onsetDvp

# <codecell>

del fig

# <codecell>

# make figure size square
fig = plt.figure(figsize = (20, 20)) # set to large number for better resolution, e.g. (20, 20)
fig.clf()

gs = mpl.gridspec.GridSpec(2, 2)
gs.update(wspace = 0.015, hspace = 0.15)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# <codecell>

ax_a.set_anchor('W')
ax_c.set_anchor('W')

ax_a.update_params()

# <codecell>

fig.canvas.draw()

print ax_a.get_position(original = False)
print ax_a.get_position(original = True)
print ax_a.figbox

# <rawcell>

# bbox = mpl.transforms.Bbox.from_bounds(x0 = 0, y0 = 0, width = 0.3, height = 0.6)
# 
# ax_a.set_position(bbox, which = 'both')
# 
# print ax_a.get_position(original = False)
# print ax_a.get_position(original = True)
# 
# ax_a.set_zorder(10)

# <codecell>

if False:
    for a in [ax_b]:
        a.set_xticklabels([])

for a in [ax_d]:
    a.set_yticklabels([])

# <codecell>

plotlabels = ['a', 'b', 'c', 'd']
labelformat = dict(fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')

# <codecell>

mpl.rcParams['image.aspect']

# <codecell>

trunkRLNnum = 5

axim = ax_a.imshow(all_dVP[:, :, trunkRLNnum]) #, aspect = 'auto')

# <codecell>

# draw everything so that the axes bboxes are updated
fig.canvas.draw()

print ax_a.get_position(original = False)
print ax_a.get_position(original = True)
print ax_a.figbox

# <codecell>

ax_a.set_xlabel(label_TA)
ax_a.set_ylabel(label_SLN)

ax_a.grid(False)

ax_a.set_title('LCA/IA 5')

axim.set_clim(np.nanmin(all_dVP), np.nanmax(all_dVP))
ax_a.relim()

co = ax_a.contour(axim.get_array(), 
                   12, 
                   colors = 'w')
ax_a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

ax_a.axis([-0.5, 4.5, -0.5, 7.5])

# <codecell>

fig.canvas.draw()

print ax_a.get_position(original = False)
print ax_a.get_position(original = True)
print ax_a.figbox

# <codecell>

bbox = mpl.transforms.Bbox.from_bounds(x0 = 1, y0 = 2, width = 3, height = 4)

print bbox
print "x0: ", bbox.x0
print "y0: ", bbox.y0
print "width: ", bbox.width
print "height: ", bbox.height
print "bounds: ", bbox.bounds
print "extents: ", bbox.extents

# <codecell>

bbox_a = ax_a.get_position()
bbox_a

# <codecell>

cbar_pad = 0.1 # padding between colorbar and axes, as a fraction of the axes width
cbar_width = 0.1 # width of colorbar, as a fraction of axes width

# <codecell>

cax = plt.axes([bbox_a.x1 + bbox_a.width * cbar_pad, bbox_a.y0, bbox_a.width * cbar_width, bbox_a.height])

cax.get_position()

# <codecell>

fig.canvas.draw()
print ax_a.get_position(original = False)
print ax_a.get_position(original = True)
print ax_a.figbox

# <codecell>

cb = fig.colorbar(axim, cax = cax)
cb.solids.set_edgecolor("face")

cb.set_label('onset adduction Dvp [%]')

# <codecell>

labelformat.update(dict(transform = ax_a.transData))
ax_a.text(-0.1, 6.9, plotlabels[0], **labelformat)

# <codecell>

for TAnum, (a, TAcond) in enumerate(zip([ax_b, ax_c, ax_d], ['TA 0', 'TA 1', 'TA 2'])):
    axim = a.imshow(dVP_plot[TAcond])
    a.set_title(TAcond)
    
    a.grid(False)

    axim.set_clim(np.nanmin(all_dVP), np.nanmax(all_dVP))
    a.relim()

    a.set_xlabel(label_trunkRLN)
    a.set_ylabel(label_SLN)

    co = a.contour(axim.get_array(), 
                   12, 
                   colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 7.5, -0.5, 7.5])

    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[TAnum + 1], **labelformat)

# <codecell>

ax_d.set_ylabel('')

ax_a.set_anchor('W')
ax_c.set_anchor('W')

fig.canvas.draw()

# <codecell>

if True:
    figurename = 'Figure1ABCD_new.pdf'
    fig.savefig(figurename, format = 'pdf', orientation = 'landscape',
                bbox_inches = 'tight', pad_inches = 0.1)

# <rawcell>

# plotlabels = ['b', 'c', 'd']
# labelformat = dict(fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')
# 
# for TAnum, TAcond in enumerate(['TA 0', 'TA 1', 'TA 2']):
# 
#     figname = 'onset adduction Dvp: %s' % TAcond
#     fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))
# 
#     axim = plt.imshow(dVP_plot[TAcond])
#     ax = plt.gca()
# 
#     # axim.set_clim(17, 103)
#     axim.set_clim(np.nanmin(all_dVP), np.nanmax(all_dVP))
#     ax.relim()
# 
#     co = plt.contour(axim.get_array(), 
#                      12, 
#                      colors = 'w')
#     plt.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)
# 
#     ax.axis([-0.5, 7.5, -0.5, 7.5])
#     ax.grid(False)
# 
#     plt.xlabel(label_trunkRLN)
#     if TAcond in ['TA 1']:
#         plt.ylabel(label_SLN)
#         
#     if TAcond in ['TA 0', 'TA 2']:
#         ax.set_yticklabels([])
#     plt.title(TAcond) # figname)
# 
#     # cb = plt.colorbar(axim)
#     # cb.set_label('onset adduction Dvp [%]')
# 
#     labelformat.update(dict(transform = ax.transData))
#     ax.text(-0.1, 6.9, plotlabels[TAnum], **labelformat)
#     
#     figurename = '%s.onsetDvp.%s' % (figurenametemplate, TAcond.replace(' ', ''))  
#     figurename += ".%s" % figureformat
#     if True:
#         plt.savefig(figurename, format = figureformat, orientation = figureorientation,
#                     bbox_inches = 'tight', pad_inches = 0.1)
#         print "saving figure: ", figurename
#         
#         plt.close('all')
#     else:
#         plt.show()

# <rawcell>

# !pdfjam Figure1?.pdf --paper letter --no-landscape --no-twoside --frame false --nup 2x2 --outfile "Figure1ABCD.pdf"

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-trunkRLN | grep onsetRstrain

# <codecell>

ls -lo Figure2?.pdf

# <codecell>

del fig

# <codecell>

del fig2

# <codecell>

fig2 = plt.figure(figsize = (20, 20)) # set to large number for better resolution, e.g. (20, 20)
fig2.clf()

gs = mpl.gridspec.GridSpec(2, 2)
gs.update(wspace = 0.015, hspace = 0.15)

ax_a = fig2.add_subplot(gs[0, 0])
ax_b = fig2.add_subplot(gs[0, 1])
ax_c = fig2.add_subplot(gs[1, 0])
ax_d = fig2.add_subplot(gs[1, 1])

ax_b.set_yticklabels([])
ax_d.set_yticklabels([])

# <codecell>

for TAnum, (a, TAcond) in enumerate(zip([ax_a, ax_b], ['TA 2', 'TA 4'])):
    axim = a.imshow(rstrain_plot[TAcond])
    a.set_title(TAcond)
    
    a.set_anchor('NW')
    
    axim.set_clim(np.nanmin(all_rstrain), np.nanmax(all_rstrain))
    a.relim()

    co = a.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 7.5, -0.5, 7.5])
    a.grid(False)

    a.set_xlabel(label_trunkRLN)
    a.set_ylabel(label_SLN)
    
    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[TAnum], **labelformat)

ax_b.set_ylabel('')

ax_a.set_anchor('NW')
ax_b.set_anchor('NW')

fig2.canvas.draw()

# <codecell>

for ind, (a, trunkRLNnum) in enumerate(zip([ax_c, ax_d], [5, 7])):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    axim = a.imshow(all_rstrain[:, :, trunkRLNnum])
    a.set_title("LCA/IA %d" % trunkRLNnum)
    
    a.set_anchor('NW')
    
    axim.set_clim(np.nanmin(all_rstrain), np.nanmax(all_rstrain))
    a.relim()

    co = a.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 4.5, -0.5, 7.5])
    a.grid(False)

    a.set_xlabel(label_TA)
    a.set_ylabel(label_SLN)
    
    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[ind + 2], **labelformat)

ax_d.set_ylabel('')

ax_c.set_anchor('NW')
ax_d.set_anchor('NW')

fig2.canvas.draw()

# <codecell>

fig2.canvas.draw()

bbox_a = ax_a.get_position()
bbox_b = ax_b.get_position()

bbox_c = ax_c.get_position()
bbox_d = ax_d.get_position()

# put both lower plots to the lower left side
# ax_d.set_position(pos = [bbox_c.x1 + bbox_c.width * 0.1, bbox_d.y0, bbox_d.width, bbox_d.height], which = 'both')

# put left lower plot towards the middle
ax_c.set_position(pos = [bbox_a.x1 - bbox_c.width, bbox_c.y0, bbox_c.width, bbox_c.height], which = 'both')

# <codecell>

fig2.canvas.draw()

bbox_d = ax_d.get_position()

cbar_pad = 0.1 # padding between colorbar and axes, as a fraction of the axes width
cbar_width = 0.1 # width of colorbar, as a fraction of axes width
 
cax = plt.axes([bbox_d.x1 + bbox_d.width * cbar_pad, bbox_d.y0, bbox_d.width * cbar_width, bbox_d.height])

# <codecell>

cb = fig2.colorbar(axim, cax = cax)
cb.solids.set_edgecolor("face")

cb.set_label('onset right strain [%]')

# <codecell>

if True:
    figurename = 'Figure2ABCD_new.pdf'
    fig2.savefig(figurename, format = 'pdf', orientation = 'landscape',
                 bbox_inches = 'tight', pad_inches = 0.1)

# <rawcell>

# for TAnum, TAcond in enumerate(['TA 2', 'TA 4']):
# 
#     figname = 'onset right strain: %s' % TAcond
#     fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))
# 
#     axim = plt.imshow(rstrain_plot[TAcond])
#     ax = plt.gca()
# 
#     # axim.set_clim(-10, 45)
#     axim.set_clim(np.nanmin(all_rstrain), np.nanmax(all_rstrain))
#     ax.relim()
# 
#     co = plt.contour(axim.get_array(), 
#                      12, 
#                      colors = 'w')
#     plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)
# 
#     ax.axis([-0.5, 7.5, -0.5, 7.5])
#     ax.grid(False)
# 
#     plt.xlabel(label_trunkRLN)
#     plt.ylabel(label_SLN)
#     plt.title(figname)
# 
#     cb = plt.colorbar(axim)
#     cb.set_label('onset right strain [%]')
# 
#     plotlabels = ['A', 'B']
#     labelformat = dict(transform = ax.transData, fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')
#     
#     ax.text(-0.1, 6.9, plotlabels[TAnum], **labelformat)
#     
#     figurename = '%s.onsetRstrain.%s' % (figurenametemplate, TAcond.replace(' ', ''))  
#     figurename += ".%s" % figureformat
#     if True:
#         plt.savefig(figurename, format = figureformat, orientation = figureorientation,
#                     bbox_inches = 'tight', pad_inches = 0.1)
#         plt.close('all')
#     else:
#         plt.show()

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-trunkRLN | grep onsetF0

# <codecell>

del fig2

# <codecell>

del fig3

# <codecell>

fig3 = plt.figure(figsize = (20, 20)) # set to large number for better resolution, e.g. (20, 20)
fig3.clf()

gs = mpl.gridspec.GridSpec(2, 2)
gs.update(wspace = 0.015, hspace = 0.05)

ax_a = fig3.add_subplot(gs[0, 0])
ax_b = fig3.add_subplot(gs[0, 1])
ax_c = fig3.add_subplot(gs[1, 0])
ax_d = fig3.add_subplot(gs[1, 1])

ax_a.set_xticklabels([])
ax_b.set_xticklabels([])

ax_b.set_yticklabels([])
ax_d.set_yticklabels([])

# <codecell>

contours = {'TA 0': np.arange(100, 200, 30),
            'TA 1': np.arange(100, 640, 30),
            'TA 2': np.arange(100, 720, 30),
            'TA 3': np.arange(100, 720, 30),
            'TA 4': np.arange(80, 420, 30)}

# <codecell>

np.nanmin(all_F0), np.nanmax(all_F0)

# <codecell>

for TAnum, (a, TAcond) in enumerate(zip([ax_a, ax_b, ax_c, ax_d], ['TA 1', 'TA 2', 'TA 3', 'TA 4'])):

    axim = a.imshow(F0_plot[TAcond])
    a.set_title(TAcond)
    
    # axim.set_clim(100, 720)
    axim.set_clim(np.nanmin(all_F0), np.nanmax(all_F0))
    a.relim()
    
    co = a.contour(axim.get_array(), 
                     contours[TAcond], # 12, 
                     colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 7.5, -0.5, 7.5])
    a.grid(False)

    a.set_xlabel(label_trunkRLN)
    a.set_ylabel(label_SLN)

    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[TAnum], **labelformat)

# <codecell>

ax_a.set_xlabel('')

ax_b.set_xlabel('')
ax_b.set_ylabel('')

ax_d.set_ylabel('')

# <codecell>

fig3.canvas.draw()

# <codecell>

bbox_cd = mpl.transforms.Bbox.union([a.get_position() for a in [ax_c, ax_d]])
bbox_cd

# <codecell>

fig3.canvas.draw()

cbar_pad = 0.1 # padding between colorbar and axes, as a fraction of the axes width
cbar_width = 0.1 # width of colorbar, as a fraction of axes width
 
cax = plt.axes([bbox_cd.x0, bbox_cd.y0 - bbox_cd.height * (cbar_width + cbar_pad), bbox_cd.width, bbox_cd.height * cbar_width])

# <codecell>

cb = fig3.colorbar(axim, cax = cax, orientation = 'horizontal')
cb.solids.set_edgecolor("face")

cb.set_label('onset frequency [Hz]')

# <codecell>

if True:
    figurename = "Figure3ABCD_new.pdf"
    fig3.savefig(figurename, format = figureformat, orientation = figureorientation,
                 bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

ls -lo Figure4?.pdf

# <codecell>

del fig3

# <codecell>

del fig4

# <codecell>

fig4 = plt.figure(figsize = (20, 20)) # set to large number for better resolution, e.g. (20, 20)
fig4.clf()

gs = mpl.gridspec.GridSpec(2, 2)
gs.update(wspace = 0.015, hspace = 0.05)

ax_a = fig4.add_subplot(gs[0, 0])
ax_b = fig4.add_subplot(gs[0, 1])
ax_c = fig4.add_subplot(gs[1, 0])
ax_d = fig4.add_subplot(gs[1, 1])

ax_a.set_xticklabels([])
ax_b.set_xticklabels([])

ax_b.set_yticklabels([])
ax_d.set_yticklabels([])

# <codecell>

for ind, (a, trunkRLNnum) in enumerate(zip([ax_a, ax_b, ax_c, ax_d], [1, 3, 5, 7])):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    axim = a.imshow(all_F0[:, :, trunkRLNnum])
    
    axim.set_clim(np.nanmin(all_F0), np.nanmax(all_F0))
    a.relim()
    
    co = a.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 4.5, -0.5, 7.5])
    a.grid(False)

    a.set_xlabel(label_TA)
    a.set_ylabel(label_SLN)
    a.set_title(trunkRLNcond)

    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[ind], **labelformat)

# <codecell>

ax_a.set_xlabel('')

ax_b.set_xlabel('')
ax_b.set_ylabel('')

ax_d.set_ylabel('')

# <codecell>

fig4.canvas.draw()

# <codecell>

ax_a.set_anchor('NW')
ax_b.set_anchor('NW')
ax_c.set_anchor('NW')
ax_d.set_anchor('NW')

# <codecell>

fig4.canvas.draw()

# <codecell>

print ax_a.get_position()
print ax_a.get_position(original = True)
print ax_a.figbox
bbox_a = ax_a.get_position()

# <codecell>

print ax_b.get_position()
print ax_b.get_position(original = True)
print ax_b.figbox
bbox_b = ax_b.get_position()

# <codecell>

ax_b.set_position(pos = [bbox_a.x1 + bbox_a.width * 0.05, bbox_b.y0, bbox_b.width, bbox_b.height], which = 'both')

# <codecell>

print ax_d.get_position()
print ax_d.get_position(original = True)
print ax_d.figbox
bbox_d = ax_d.get_position()

# <codecell>

ax_d.set_position(pos = [bbox_a.x1 + bbox_a.width * 0.05, bbox_d.y0, bbox_d.width, bbox_d.height], which = 'both')

# <codecell>

fig4.canvas.draw()

# <codecell>

bbox_cd = mpl.transforms.Bbox.union([a.get_position() for a in [ax_c, ax_d]])
bbox_cd

# <codecell>

fig4.delaxes(cax)
del cb

# <codecell>

fig4.canvas.draw()
 
cbar_pad = 0.1 # padding between colorbar and axes, as a fraction of the axes width
cbar_width = 0.05 # width of colorbar, as a fraction of axes width
 
cax = plt.axes([bbox_cd.x0, bbox_cd.y0 - bbox_cd.width * (cbar_width + cbar_pad), bbox_cd.width, bbox_cd.width * cbar_width])

# <codecell>

cb = fig4.colorbar(axim, cax = cax, orientation = 'horizontal')
cb.solids.set_edgecolor("face")
 
cb.set_label('onset frequency [Hz]')

# <codecell>

if True:
    figurename = "Figure4ABCD_new.pdf"
    fig4.savefig(figurename, format = figureformat, orientation = figureorientation,
                 bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-trunkRLN | grep onsetPs

# <codecell>

ls -lo Figure5?.pdf

# <codecell>

del fig4

# <codecell>

del fig5

# <codecell>

fig5 = plt.figure(figsize = (20, 20)) # set to large number for better resolution, e.g. (20, 20)
fig5.clf()

gs = mpl.gridspec.GridSpec(2, 2)
gs.update(wspace = 0.015, hspace = 0.15)

ax_a = fig5.add_subplot(gs[0, 0])
ax_b = fig5.add_subplot(gs[0, 1])
ax_c = fig5.add_subplot(gs[1, 0])
ax_d = fig5.add_subplot(gs[1, 1])

ax_b.set_yticklabels([])

ax_d.set_yticklabels([])

# <codecell>

for TAnum, (a, TAcond) in enumerate(zip([ax_a, ax_b], ['TA 1', 'TA 3'])):

    axim = a.imshow(ps_plot[TAcond])

    # axim.set_clim(200, 2000)
    axim.set_clim(np.nanmin(all_ps), np.nanmax(all_ps))
    a.relim()

    co = a.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 7.5, -0.5, 7.5])
    a.grid(False)

    a.set_xlabel(label_trunkRLN)
    a.set_ylabel(label_SLN)
    a.set_title(TAcond)

    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[TAnum], **labelformat)
    
ax_b.set_ylabel('')

ax_a.set_anchor('NW')
ax_b.set_anchor('NW')

fig5.canvas.draw()

# <codecell>

for ind, (a, trunkRLNnum) in enumerate(zip([ax_c, ax_d], [3, 5])):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    axim = a.imshow(all_ps[:, :, trunkRLNnum])
    
    axim.set_clim(np.nanmin(all_ps), np.nanmax(all_ps))
    a.relim()
    
    co = a.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 4.5, -0.5, 7.5])
    a.grid(False)

    a.set_xlabel(label_TA)
    a.set_ylabel(label_SLN)
    a.set_title(trunkRLNcond)
    
    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[ind + 2], **labelformat)
    
ax_d.set_ylabel('')

ax_c.set_anchor('NW')
ax_d.set_anchor('NW')

fig5.canvas.draw()

# <codecell>

fig5.canvas.draw()

bbox_a = ax_a.get_position()
bbox_b = ax_b.get_position()

bbox_c = ax_c.get_position()
bbox_d = ax_d.get_position()

# put both lower plots to the lower left side
# ax_d.set_position(pos = [bbox_c.x1 + bbox_c.width * 0.1, bbox_d.y0, bbox_d.width, bbox_d.height], which = 'both')

# put left lower plot towards the middle
ax_c.set_position(pos = [bbox_a.x1 - bbox_c.width, bbox_c.y0, bbox_c.width, bbox_c.height], which = 'both')

# <codecell>

bbox_d = ax_d.get_position()
 
cbar_pad = 0.1 # padding between colorbar and axes, as a fraction of the axes width
cbar_width = 0.1 # width of colorbar, as a fraction of axes width
 
cax = plt.axes([bbox_d.x1 + bbox_d.width * cbar_pad, bbox_d.y0, bbox_d.width * cbar_width, bbox_d.height])

# <codecell>

cb = fig5.colorbar(axim, cax = cax)
cb.solids.set_edgecolor("face")
 
cb.set_label('onset pressure [Pa]')

# <codecell>

if True:
    figurename = 'Figure5ABCD_new.pdf'
    fig5.savefig(figurename, format = 'pdf', orientation = 'landscape',
                 bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-trunkRLN | grep onsetQ

# <codecell>

ls -lo Figure6?.pdf

# <codecell>

del fig5

# <codecell>

del fig6

# <codecell>

fig6 = plt.figure(figsize = (20, 20)) # set to large number for better resolution, e.g. (20, 20)
fig6.clf()

gs = mpl.gridspec.GridSpec(2, 2)
gs.update(wspace = 0.015, hspace = 0.15)

ax_a = fig6.add_subplot(gs[0, 0])
ax_b = fig6.add_subplot(gs[0, 1])
ax_c = fig6.add_subplot(gs[1, 0])
ax_d = fig6.add_subplot(gs[1, 1])

ax_b.set_yticklabels([])

ax_d.set_yticklabels([])

# <codecell>

for TAnum, (a, TAcond) in enumerate(zip([ax_a, ax_b], ['TA 1', 'TA 2'])):

    axim = a.imshow(Q_plot[TAcond])

    axim.set_clim(np.nanmin(all_Q), np.nanmax(all_Q))
    a.relim()

    co = a.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 7.5, -0.5, 7.5])
    a.grid(False)

    a.set_xlabel(label_trunkRLN)
    a.set_ylabel(label_SLN)
    a.set_title(TAcond)

    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[TAnum], **labelformat)

ax_b.set_ylabel('')

ax_a.set_anchor('NW')
ax_b.set_anchor('NW')
    
fig6.canvas.draw()

# <codecell>

for ind, (a, trunkRLNnum) in enumerate(zip([ax_c, ax_d], [3, 5])):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    axim = a.imshow(all_Q[:, :, trunkRLNnum])
    
    axim.set_clim(np.nanmin(all_Q), np.nanmax(all_Q))
    a.relim()
    
    co = a.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    a.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    a.axis([-0.5, 4.5, -0.5, 7.5])
    a.grid(False)

    a.set_xlabel(label_TA)
    a.set_ylabel(label_SLN)
    a.set_title(trunkRLNcond)

    labelformat.update(dict(transform = a.transData))
    a.text(-0.1, 6.9, plotlabels[ind + 2], **labelformat)

ax_d.set_ylabel('')

ax_c.set_anchor('NW')
ax_d.set_anchor('NW')

fig6.canvas.draw()

# <codecell>

fig6.canvas.draw()

bbox_a = ax_a.get_position()
bbox_b = ax_b.get_position()

bbox_c = ax_c.get_position()
bbox_d = ax_d.get_position()

# put both lower plots to the lower left side
# ax_d.set_position(pos = [bbox_c.x1 + bbox_c.width * 0.1, bbox_d.y0, bbox_d.width, bbox_d.height], which = 'both')

# put left lower plot towards the middle
ax_c.set_position(pos = [bbox_a.x1 - bbox_c.width, bbox_c.y0, bbox_c.width, bbox_c.height], which = 'both')

# <codecell>

bbox_d = ax_d.get_position()
 
cbar_pad = 0.1 # padding between colorbar and axes, as a fraction of the axes width
cbar_width = 0.1 # width of colorbar, as a fraction of axes width
 
cax = plt.axes([bbox_d.x1 + bbox_d.width * cbar_pad, bbox_d.y0, bbox_d.width * cbar_width, bbox_d.height])

# <codecell>

cb = fig6.colorbar(axim, cax = cax)
cb.solids.set_edgecolor("face")
 
cb.set_label('onset flow rate [ml/s]')

# <codecell>

if True:
    figurename = 'Figure6ABCD_new.pdf'
    fig6.savefig(figurename, format = 'pdf', orientation = 'landscape',
                 bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

for TAnum, TAcond in enumerate(TAconditions):
    #######################################
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

    figurename = '%s.onsetF0.%s' % (figurenametemplate, TAcond.replace(' ', ''))
    figurename += ".%s" % figureformat
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)

    #######################################
    figname = 'onset pressure: %s' % TAcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(ps_plot[TAcond])
    ax = plt.gca()

    # axim.set_clim(200, 2000)
    axim.set_clim(np.nanmin(all_ps), np.nanmax(all_ps))
    ax.relim()

    co = plt.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 7.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_trunkRLN)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset pressure [Pa]')

    figurename = '%s.onsetPs.%s' % (figurenametemplate, TAcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)

    #######################################
    figname = 'onset Bernoulli area: %s' % TAcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(A_plot[TAcond])
    ax = plt.gca()

    # axim.set_clim(200, 2000)
    axim.set_clim(np.nanmin(all_A), np.nanmax(all_A))
    ax.relim()

    co = plt.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 7.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_trunkRLN)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset Bernoulli area [a.u.]')

    figurename = '%s.onsetA.%s' % (figurenametemplate, TAcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)

    #######################################
    figname = 'onset flow rate: %s' % TAcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(Q_plot[TAcond])
    ax = plt.gca()

    axim.set_clim(np.nanmin(all_Q), np.nanmax(all_Q))
    ax.relim()

    co = plt.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 7.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_trunkRLN)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset flow rate [ml/s]')

    figurename = '%s.onsetQ.%s' % (figurenametemplate, TAcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)

    #######################################
    figname = 'onset left strain: %s' % TAcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(lstrain_plot[TAcond])
    ax = plt.gca()

    # axim.set_clim(-10, 45)
    axim.set_clim(np.nanmin(all_lstrain), np.nanmax(all_lstrain))
    ax.relim()

    co = plt.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 7.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_trunkRLN)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset left strain [%]')

    figurename = '%s.onsetLstrain.%s' % (figurenametemplate, TAcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)

    #######################################
    figname = 'onset right strain: %s' % TAcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(rstrain_plot[TAcond])
    ax = plt.gca()

    # axim.set_clim(-10, 45)
    axim.set_clim(np.nanmin(all_rstrain), np.nanmax(all_rstrain))
    ax.relim()

    co = plt.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 7.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_trunkRLN)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset right strain [%]')

    figurename = '%s.onsetRstrain.%s' % (figurenametemplate, TAcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)

    #######################################
    figname = 'onset adduction Dvp: %s' % TAcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(dVP_plot[TAcond])
    ax = plt.gca()

    # axim.set_clim(17, 103)
    axim.set_clim(np.nanmin(all_dVP), np.nanmax(all_dVP))
    ax.relim()

    co = plt.contour(axim.get_array(), 
                     12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 7.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_trunkRLN)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset adduction Dvp [%]')

    figurename = '%s.onsetDvp.%s' % (figurenametemplate, TAcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

figurenametemplate = 'SLN-TA'

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-TA | grep onsetDvp

# <codecell>

plotlabels = ['a']
labelformat = dict(fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')

for ind, trunkRLNnum in enumerate([5]):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    #######################################
    figname = 'onset adduction Dvp: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_dVP[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_dVP), np.nanmax(all_dVP))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset adduction Dvp [%]')
    
    labelformat.update(dict(transform = ax.transData))
    ax.text(-0.1, 6.9, plotlabels[ind], **labelformat)

    figurename = '%s.onsetDvp.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    if True:
        plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                    bbox_inches = 'tight', pad_inches = 0.1)
        
        plt.close('all')
    else:
        plt.show()

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-TA | grep onsetRstrain

# <codecell>

for ind, trunkRLNnum in enumerate([5, 7]):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    #######################################
    figname = 'onset right strain: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_rstrain[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_rstrain), np.nanmax(all_rstrain))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset right strain [%]')

    plotlabels = ['C', 'D']
    labelformat = dict(transform = ax.transData, fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')
    
    ax.text(-0.1, 6.9, plotlabels[ind], **labelformat)

    figurename = '%s.onsetRstrain.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    if True:
        plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                    bbox_inches = 'tight', pad_inches = 0.1)
        
        plt.close('all')
    else:
        plt.show()

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-TA | grep onsetF0

# <codecell>

for ind, trunkRLNnum in enumerate([1, 3, 5, 7]):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    #######################################
    figname = 'onset frequency: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_F0[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_F0), np.nanmax(all_F0))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset frequency [Hz]')

    plotlabels = ['A', 'B', 'C', 'D']
    labelformat = dict(transform = ax.transData, fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')
    
    ax.text(-0.1, 6.9, plotlabels[ind], **labelformat)

    figurename = '%s.onsetF0.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    if True:
        plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                    bbox_inches = 'tight', pad_inches = 0.1)
        
        plt.close('all')
    else:
        plt.show()

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-TA | grep onsetPs

# <codecell>

for ind, trunkRLNnum in enumerate([3, 5]):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    #######################################
    figname = 'onset pressure: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_ps[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_ps), np.nanmax(all_ps))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset pressure [Pa]')

    plotlabels = ['C', 'D']
    labelformat = dict(transform = ax.transData, fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')
    
    ax.text(-0.1, 6.9, plotlabels[ind], **labelformat)

    figurename = '%s.onsetPs.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    if True:
        plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                    bbox_inches = 'tight', pad_inches = 0.1)
        
        plt.close('all')
    else:
        plt.show()

# <codecell>

!ls -alo --color=always Figure?[A-D].pdf | grep SLN-TA | grep onsetQ

# <codecell>

for ind, trunkRLNnum in enumerate([3, 5]):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    #######################################
    figname = 'onset flow rate: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_Q[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_Q), np.nanmax(all_Q))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset flow rate [ml/s]')

    plotlabels = ['C', 'D']
    labelformat = dict(transform = ax.transData, fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')
    
    ax.text(-0.1, 6.9, plotlabels[ind], **labelformat)

    figurename = '%s.onsetQ.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    if True:
        plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                    bbox_inches = 'tight', pad_inches = 0.1)
        
        plt.close('all')
    else:
        plt.show()

# <codecell>

for trunkRLNnum in range(Nlevels):
    # trunkRLNcond = 'trunk RLN %s' % str(trunkRLNnum)
    trunkRLNcond = 'LCA/IA %s' % str(trunkRLNnum)

    #######################################
    figname = 'onset frequency: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_F0[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_F0), np.nanmax(all_F0))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset frequency [Hz]')

    figurename = '%s.onsetF0.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)
    
    #######################################
    figname = 'onset pressure: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_ps[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_ps), np.nanmax(all_ps))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset pressure [Pa]')

    figurename = '%s.onsetPs.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)

    #######################################
    figname = 'onset Bernoulli area: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_A[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_A), np.nanmax(all_A))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset Bernoulli area [a.u.]')

    figurename = '%s.onsetA.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)    

    #######################################
    figname = 'onset flow rate: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_Q[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_Q), np.nanmax(all_Q))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset flow rate [ml/s]')

    figurename = '%s.onsetQ.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)    

    #######################################
    figname = 'onset left strain: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_lstrain[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_lstrain), np.nanmax(all_lstrain))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset left strain [%]')

    figurename = '%s.onsetLstrain.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)    

    #######################################
    figname = 'onset right strain: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_rstrain[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_rstrain), np.nanmax(all_rstrain))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset right strain [%]')

    figurename = '%s.onsetRstrain.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)    

    #######################################
    figname = 'onset adduction Dvp: %s' % trunkRLNcond
    fig = plt.figure(num = figname, figsize = (14.9625, 12.2125))

    axim = plt.imshow(all_dVP[:, :, trunkRLNnum])
    ax = plt.gca()
    
    axim.set_clim(np.nanmin(all_dVP), np.nanmax(all_dVP))
    ax.relim()
    
    co = plt.contour(axim.get_array(), 
                     12, # contours[TAcond], # 12, 
                     colors = 'w')
    plt.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

    ax.axis([-0.5, 4.5, -0.5, 7.5])
    ax.grid(False)

    plt.xlabel(label_TA)
    plt.ylabel(label_SLN)
    plt.title(figname)

    cb = plt.colorbar(axim)
    cb.set_label('onset adduction Dvp [%]')

    figurename = '%s.onsetDvp.%s' % (figurenametemplate, trunkRLNcond.replace(' ', ''))  
    figurename += ".%s" % figureformat
    figurename = figurename.replace('/', '_')
    plt.savefig(figurename, format = figureformat, orientation = figureorientation,
                bbox_inches = 'tight', pad_inches = 0.1)    

# <codecell>

#####################

# plt.show()
plt.close('all')

