# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Create graphs evaluating phase and mucosal asymmetry vs stimulation levels of RLN and SLN in the 4/4/12 RLN Asymmetry.
# 
# Tables with the phase and mucosal asymmetries on Tabs 6 & 7.
# 
# Y:\Chhetri_R01_Projects\RLN asymmetry
# 
# xls file: RLN Asymmetry_4.4.12_FINAL

# <codecell>

%matplotlib inline

# <codecell>

%config InlineBackend.close_figures = False
%config InlineBackend

# <codecell>

import sys, glob, os
import numpy as np

import matplotlib as mpl
# mpl.use('module://IPython.zmq.pylab.backend_inline')
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['image.cmap'] = 'jet'

sys.path.append("/extra/InVivoDog/python/cine/tools")

from dogdata import DogData

import xlrd

# <codecell>

mpl.rcParams['savefig.dpi']

# <codecell>

colorlist = ['c', 'b', 'g', 'm', 'r']
cmap = mpl.colors.ListedColormap(colorlist)

symblist = ['*', '+', '^', 'v', 'o']

# <codecell>

excel_dir = "/extra/InVivoDog/Dinesh/RLN_asymmetry"

# <codecell>

glob.glob(os.path.join(excel_dir, '*.xlsx'))

# <codecell>

ls -alot $excel_dir

# <markdowncell>

# Load onset data (onset time, F0 at onset) from Excel file
# =========================================================

# <codecell>

excel_onsetfile = 'RLN Asymmetry_4.4.12_FINAL.xlsx'
# excel_onsetfile = 'RLN Asymmetry_2.15.12_FINAL.xlsx'

# <codecell>

excel_filename = os.path.join(excel_dir, excel_onsetfile)
print "opening Excel file: ", excel_filename
print
book = xlrd.open_workbook(filename = excel_filename)

print 'sheets: ', book.sheet_names()
print
for sheet in book.sheets():
    print "%d: %s\t\t: rows: %d\t, columns: %d" % (sheet.number, sheet.name, sheet.nrows, sheet.ncols)

# <codecell>

print sheet.row_values(rowx = 3)
col_phase = sheet.row_values(rowx = 3).index('Phase')
print "col_phase = ", col_phase

# <markdowncell>

# Get pressure and flow data to interpolate from onset data
# =========================================================

# <codecell>

datadir = "/extra/InVivoDog/Dinesh/RLN_asymmetry/InVivoDog_2012_04_04/"
print "%s exists?" % datadir, os.path.isdir(datadir)

datafiles = {"SLN 0": "asymmetricRLN NoSLN Wed Apr 04 2012 14 51 58.hdf5",
             "SLN 1": "asymmetricRLN ThresholdSLN_condition01 Wed Apr 04 2012 14 59 14.hdf5",
             "SLN 2": "asymmetricRLN SLN_condition02 Wed Apr 04 2012 15 09 45.hdf5",
             "SLN 3": "asymmetricRLN SLN_condition03 Wed Apr 04 2012 15 17 06.hdf5",
             "SLN 4": "asymmetricRLN MaxSLN_condition04 Wed Apr 04 2012 15 25 32.hdf5"}

SLNconditions = sorted(datafiles.keys())

SLNconditions_names = ['No SLN', 'Threshold SLN', 'SLN 2', 'SLN 3', 'Max SLN']

# <codecell>

F0 = {} # onset frequency in Hz
onset_time = {} # onset time in milliseconds
phase_asym = {}
mucosal_asym = {}

# <codecell>

for SLNnum, (SLNcond, SLNcondname) in enumerate(zip(SLNconditions, SLNconditions_names)):

    dogdata = DogData(datadir = datadir, datafile = datafiles[SLNcond])
    print "working on: ", dogdata.datafilename
    
    Nrecnums = dogdata.Nrecnums
    
    sheet_SLNcond = book.sheet_by_name(SLNcondname)
    print "getting data from Excel sheet: ", sheet_SLNcond.name
    print
    
    # prepare lists to hold these numbers for the different SLN conditions
    onset_time[ SLNcond ] = []
    F0[ SLNcond ] = []
    phase_asym[ SLNcond] = []
    mucosal_asym[ SLNcond ] = []
    
    for recnum in range(Nrecnums):
        # Onset sample, Onset time, #Peaks, Time (samples)
        onset, dummy, Npeaks, T = sheet_SLNcond.row_values(4 + recnum, start_colx = 1, end_colx = 5) # exclusive end_colx
        phase, mucosal = sheet_SLNcond.row_values(4 + recnum, start_colx = col_phase, end_colx = col_phase + 2)
        
        if onset in ['NP', '', ' ']:
            onset = np.nan
        else:
            onset = int(onset)
            
        # onset time in samples: sampling rate of audio signal: 50 KHz
        onset_time[ SLNcond ].append( onset / 50.0 ) # onset time in milliseconds
        
        if T in ['', ' ']:
            T = np.nan
        else:
            T = int(T)
            
        F0[ SLNcond ].append( Npeaks * 50.0 / float(T) * 1000.0 )
        
        phase = phase.lower()
        mucosal = mucosal.lower()
        
        if phase in ['', 'n/a', '?']:
            phase = np.nan
        else:
            if phase == 'l': phase = -1.0
            if phase == 's': phase = 0.0
            if phase == 'r': phase = 1.0
                
        phase_asym[ SLNcond ].append(phase)
        
        if mucosal in ['', 'n/a', '?']:
            mucosal = np.nan
        else:
            if mucosal == 'l': mucosal = -1.0
            if mucosal == 's': mucosal = 0.0
            if mucosal == 'r': mucosal = 1.0
                
        mucosal_asym[ SLNcond ].append(mucosal)

# <codecell>

print sorted(F0)
print sorted(onset_time)
print sorted(phase_asym)
print sorted(mucosal_asym)

# <codecell>

a_F0 = np.array([F0[item] for item in sorted(F0)]).T
a_onset = np.array([onset_time[item] for item in sorted(onset_time)]).T

a_phase = np.array([phase_asym[item] for item in sorted(phase_asym)]).T
a_mucosal = np.array([mucosal_asym[item] for item in sorted(mucosal_asym)]).T

# <codecell>

a_onset.shape

# <codecell>

assert a_onset.shape == (64, 5)

# <codecell>

ps_onset = np.ones(a_onset.shape) * np.nan
Q_onset = np.ones(a_onset.shape) * np.nan

ps_noonset = np.ones(a_onset.shape) * np.nan
Q_noonset = np.ones(a_onset.shape) * np.nan

# Bernoulli equivalent area A proportional to Q / sqrt(p)
A_onset = np.ones(a_onset.shape) * np.nan
A_noonset = np.ones(a_onset.shape) * np.nan

# <codecell>

zip(SLNconditions, SLNconditions_names)

# <codecell>

for SLNnum, (SLNcond, SLNcondname) in enumerate(zip(SLNconditions, SLNconditions_names)):

    dogdata = DogData(datadir = datadir, datafile = datafiles[SLNcond])
    print "working on: ", dogdata.datafilename
    dogdata.get_all_data()

    time = dogdata.time_psQ * 1000.0 # onset time is given in milliseconds
    
    for stimind, datarow in enumerate(dogdata.allps):
        if not np.isnan(a_onset[stimind, SLNnum]):
            # linear interpolation
            ps_onset[stimind, SLNnum] = np.interp(a_onset[stimind, SLNnum], time, datarow)
        else:
            ps_noonset[stimind, SLNnum] = datarow.max()

    for stimind, datarow in enumerate(dogdata.allQ):
        if not np.isnan(a_onset[stimind, SLNnum]):
            Q_onset[stimind, SLNnum] = np.interp(a_onset[stimind, SLNnum], time, datarow)
        else:
            # the maximum flow rate should always be constant as set by the flow controller for the max of the flow ramp
            Q_noonset[stimind, SLNnum] = datarow.max()

A_onset = Q_onset / np.sqrt(ps_onset)
A_noonset = Q_noonset / np.sqrt(ps_noonset)

# <codecell>

# onset, dummy, dummy, T = sheet_SLNcond.row_values((4 - 1) + 6, start_colx = 1, end_colx = 5)

# <markdowncell>

# Get landmark clicks
# ===================

# <codecell>

# landmarks at rest and onset clicked by Eli
# if onset didn't occur, used the last frame in recording
# see strainanalysis.py
landmarkdir = "/extra/InVivoDog/python/cine/results_save/04_04_2012_asymmetricRLN/"

print "%s exists? " % landmarkdir, os.path.isdir(landmarkdir)

clickfiles = sorted(glob.glob(os.path.join(landmarkdir, '*.npz')))

# <codecell>

clickfiles[:5]

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

strains = {}

for SLNnum, SLNcond in enumerate(SLNconditions):
    strains[ SLNcond ] = []

    SLNcond_name = datafiles[SLNcond].split('Wed')[0].replace(' ', '_')
    print "SLNcond_name, SLNnum: ", (SLNcond_name, SLNnum)

    SLNcond_landmarkfiles = sorted(
        [item for item in clickfiles if os.path.basename(item).startswith(SLNcond_name)])
    
    # debugging:
    # strains[SLNcond] = SLNcond_landmarkfiles

    for clickfile in SLNcond_landmarkfiles:
        data = np.load(clickfile)

        if data['TAconditionindex'] != SLNnum:
            print "Error: wrong SLN condition index"
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

        strains[ SLNcond ].append([leftstrain, rightstrain, dVPrel])

# <codecell>

Nlevels = dogdata.Nlevels
print dogdata.nervenamelist
# need left RLN versus right RLN
rellevels = dogdata.a_rellevels[:, [2, 3]]
# dogdata.a_rellevels

# <codecell>

# var_strings and data_vars need to be sorted in the same way
# maybe better: [('F0', a_F0), ('ps', ps_onset), ('Q', Q_onset), ('A', A_onset), 
# ('lstrain', None), ('rstrain', None), ('dVP', None)]
var_strings = ['F0', 'ps', 'Q', 'A', 'phase', 'mucosal', 'lstrain', 'rstrain', 'dVP']
data_vars   = [a_F0, ps_onset, Q_onset, A_onset, a_phase, a_mucosal]

# No Phonation conditions
np_strings = ['ps', 'Q', 'A', 'lstrain', 'rstrain', 'dVP']
data_np    = [ps_noonset, Q_noonset, A_noonset]

# <codecell>

var_plot = dict()
var_np_plot = dict()

for var in var_strings:
    var_plot[var] = np.nan * np.ones( (Nlevels, Nlevels, len(SLNconditions) ) ) # leftRLNlevel, rightRLNlevel, SLNlevel
    
for var in np_strings:
    var_np_plot[var] = np.nan * np.ones( (Nlevels, Nlevels, len(SLNconditions) ) )

# <codecell>

coord_plot = dict()

coord_strings = ['left RLN', 'right RLN', 'SLN']

for coord in coord_strings:
    coord_plot[coord] = np.nan * np.ones( (Nlevels, Nlevels, len(SLNconditions) ) )

for SLNnum, SLNcond in enumerate(SLNconditions):
    coord_plot['SLN'][:, :, SLNnum] = SLNnum

    for stimind, (leftRLNlevel, rightRLNlevel) in enumerate(rellevels):
        # leftRLN: row index, i.e. y-axis in plot
        # rightRLN: column index, i.e. x-axis in plot
        
        coord_plot['left RLN'][leftRLNlevel, rightRLNlevel, SLNnum] = leftRLNlevel
        coord_plot['right RLN'][leftRLNlevel, rightRLNlevel, SLNnum] = rightRLNlevel

# <codecell>

for SLNnum, SLNcond in enumerate(SLNconditions):

    for stimind, (leftRLNlevel, rightRLNlevel) in enumerate(rellevels):
        # leftRLN: row index, i.e. y-axis in plot
        # rightRLN: column index, i.e. x-axis in plot
        
        for var, dat in zip(var_strings, data_vars):
            var_plot[var][leftRLNlevel, rightRLNlevel, SLNnum] = dat[stimind, SLNnum]
            
        # e.g.: F0_plot[SLNcond][leftRLNlevel, rightRLNlevel] = a_F0[stimind, SLNnum]
        
        for var, dat in zip(np_strings, data_np):
            var_np_plot[var][leftRLNlevel, rightRLNlevel, SLNnum] = dat[stimind, SLNnum]

        # last three variables: lstrain, rstrain, dVP
        for varnum, var in enumerate(var_strings[-3:]):
            var_plot[var][leftRLNlevel, rightRLNlevel, SLNnum] = strains[SLNcond][stimind][varnum]
        
        # e.g.: lstrain_plot[TAcond][SLNlevel, RLNlevel] = strains[SLNcond][stimind][0]

            if np.isnan(var_plot['F0'][leftRLNlevel, rightRLNlevel, SLNnum]):
                var_plot[var][leftRLNlevel, rightRLNlevel, SLNnum] = np.nan
                # e.g.: lstrain_plot[TAcond][SLNlevel, RLNlevel] = np.nan
                
                var_np_plot[var][leftRLNlevel, rightRLNlevel, SLNnum] = strains[SLNcond][stimind][varnum]

# <codecell>

var_plot.keys()

# <codecell>

var_labels = ['onset frequency [Hz]', 'onset pressure [Pa]', 'onset flow rate [ml/s]', 'onset Bernoulli area [a.u.]',
              'phase lead', 'mucosal wave amplitude',
              'left strain [%]', 'right strain [%]', 'Dvp [%]']

label_dict = dict(zip(var_strings, var_labels))
label_dict

# <codecell>

from mpl_toolkits.mplot3d import Axes3D

# <codecell>

coord_plot.keys()

# <codecell>

# help Axes3D.scatter

# <codecell>

# help a.contourf3D

# <codecell>

f = plt.figure()

a = f.add_subplot(111, projection = '3d')

aa = a.scatter(coord_plot['left RLN'].ravel(), coord_plot['right RLN'].ravel(), coord_plot['SLN'].ravel(),
          s = var_plot['F0'].ravel()**2 * 0.01,
          c = var_plot['F0'].ravel(),
          # alpha = 0.8, 
          edgecolors = 'None', cmap = cmap)

a.set_xlabel('left RLN')
a.set_ylabel('right RLN')
a.set_zlabel('SLN')

plt.colorbar(aa)

plt.show()

# <codecell>

SLNconditions

# <codecell>

print sorted(var_plot.keys())
print sorted(var_np_plot.keys())

# <codecell>

plotlabels = ['a', 'b']
labelformat = dict(fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none')

symmetry_cmap = mpl.colors.ListedColormap(['b', 'g', 'r'])

# <codecell>

varall_plot = dict()

for plotvar in var_strings:
    try:
        varall_plot[plotvar] = np.where(np.isnan(var_plot[plotvar]), var_np_plot[plotvar], var_plot[plotvar])
    except:
        varall_plot[plotvar] = var_plot[plotvar]

# <codecell>

varall_plot['Q'].max()

# <codecell>

def make_dual_plot(plotvar = 'F0', figsize = (20, 10)):
    
    colorbarlabel = label_dict[plotvar]
    
    fig = plt.figure(figsize = figsize) 
    #fig, ax =  plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
    fig.clf()
    
    gs = mpl.gridspec.GridSpec(1, 2)
    gs.update(wspace = 0.015, hspace = 0.15)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    
    ax_a.set_anchor('W')
    ax_b.set_anchor('W')
    
    ax_a.update_params()
    
    fig.canvas.draw()
    
    # only use SLN conditions 0 and 2
    SLNselection = [0, 2]
    for ax, SLNnum in zip([ax_a, ax_b], SLNselection):
        if plotvar == '':
            var = var_plot[plotvar]
        else:
            var = varall_plot[plotvar]
        
        minval = np.nanmin(var[:, :, SLNselection])
        maxval = np.nanmax(var[:, :, SLNselection])
        
        print "minval, maxval = ", minval, maxval
            
        # minval = min([minval, minvalnp])
        # maxval = max([maxval, maxvalnp])

        ax.set_title(SLNconditions[SLNnum])
        
        if plotvar in ['phase', 'mucosal']:
            axim = ax.imshow(var[:, :, SLNnum], cmap = symmetry_cmap)
            
            for xmaj in ax.xaxis.get_majorticklocs():
                ax.axvline(x = xmaj + 0.5, ls = ':', color = 'k', lw = 1, marker = 'None')

            for ymaj in ax.yaxis.get_majorticklocs():
                ax.axhline(y = ymaj + 0.5, ls = ':', color = 'k', lw = 1, marker = 'None')
                
            ax.plot([-0.5, 7.5], [0.5, 8.5], 'w-', marker = 'None', zorder = 1000)
        else:
            axim = ax.imshow(var[:, :, SLNnum])

        axim.set_clim(minval, maxval)
        ax.relim()
        
        if plotvar not in ['phase', 'mucosal']:
            co = ax.contour(axim.get_array(),
                                   12,
                                   colors = 'w')
            ax.clabel(co, fmt = '%.0f', fontsize = 20, inline = True)
        
        ax.plot([-0.5, 7.5], [-0.5, 7.5], 'k-')
        
        np_rows, np_cols = np.where(np.isnan(var_plot[plotvar][:, :, SLNnum]))
        ax.plot(np_cols, np_rows, 'k*', ms = 20, mfc = 'None', mec = 'black', zorder = 1000)
        
        ax.axis([-0.5, 7.5, -0.5, 7.5])
        
        ax.set_xlabel('right RLN')
        if ax == ax_a:
            ax.set_ylabel('left RLN')
        ax.grid(False)
        
        if ax == ax_b:
            ax.set_yticklabels([])
            # cb = fig.colorbar(axim, ax = ax)
            # cb.set_label(colorbarlabel)
        
        labelformat.update(dict(transform = ax.transData))
        if ax == ax_a:
            ax.text(-0.1, 6.9, plotlabels[0], **labelformat)
        if ax == ax_b:
            ax.text(-0.1, 6.9, plotlabels[1], **labelformat)
        
    fig.canvas.draw()
        
    bbox_a = ax_a.get_position()
    bbox_b = ax_b.get_position()
    
    cbar_pad = 0.05
    cbar_width = 0.075
    
    cax = plt.axes([bbox_b.x1 + bbox_b.width * cbar_pad, bbox_b.y0,
                    bbox_b.width * cbar_width, bbox_b.height])

    # use the last axim image, as both are normalized to the same range clim
    cb = fig.colorbar(axim, cax = cax)
    cb.solids.set_edgecolor('face')
    cb.set_label(colorbarlabel)
    
    if plotvar in ['phase', 'mucosal']:
        ax_a.grid(False)
        ax_b.grid(False)
        
        cb.set_ticks([-0.66, 0, 0.66])
        cb.set_ticklabels(['left', 'symmetric', 'right'])
        
        ytl = cb.ax.get_yticklabels()
        for text in ytl:
            text.set_rotation(90)
    
    figurename = 'RLN_Asymmetry_%s.pdf' % plotvar
    
    fig.savefig(figurename, orientation = 'landscape', dpi = 300,
                bbox_inches = 'tight', pad_inches = 0.1)
    
    figurename = figurename.replace('.pdf', '.eps')
    fig.savefig(figurename, orientation = 'portrait', dpi = 50,
                bbox_inches = 'tight', pad_inches = 0.1)
    
    return fig
    
    # plt.show()

# <codecell>

var_strings

# <codecell>

np_strings

# <codecell>

fig = make_dual_plot(plotvar = 'mucosal')

# <codecell>

for plotvar in var_strings:
    print plotvar
    plt.close('all')
    fig = make_dual_plot(plotvar = plotvar)

# <codecell>

def make_left_right_plot(plotvar = 'F0', uniformscale = True, savefigure = False, showfigure = False, figsize = (15, 12)):
    if figsize == None:
        figsize = mpl.rcParams['figure.figsize']

    colorbarlabel = label_dict[plotvar]

    for SLNnum, SLNcond in enumerate(SLNconditions):
        fig, ax = plt.subplots(figsize = figsize)

        ax.set_title(SLNcond)

        axim = ax.imshow(var_plot[plotvar][:, :, SLNnum])

        # scale all axis with respect to global min and max
        # uniform_scale = False
        if uniformscale:
            axim.set_clim(np.nanmin(var_plot[plotvar]), np.nanmax(var_plot[plotvar]))
            ax.relim()

        co = ax.contour(axim.get_array(), 
                        6, 
                        colors = 'w')
        ax.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

        ax.axis([-0.5, 7.5, -0.5, 7.5])

        ax.set_ylabel('left RLN')

        ax.grid(False)

        if True or SLNnum == 0:
            cb = fig.colorbar(axim, ax = ax)
            cb.set_label(colorbarlabel)

        if True or SLNcond == SLNconditions[-1]:
            ax.set_xlabel('right RLN')

        figname = "left-rightRLN.%s.%s.pdf" % ( plotvar, SLNcond.replace(' ', '_') )

        if uniformscale:
            figname = figname.replace('.pdf', '.normalized.pdf')

        if savefigure:
            print "saving: ", figname
            fig.savefig(figname, orientation = 'landscape', dpi = 300,
                        papertype = 'letter', format = 'pdf',
                        bbox_inches = 'tight', pad_inches = 0.1)


    if showfigure:
        plt.show()
    else:
        plt.close('all')

# <codecell>

var_strings

# <codecell>

make_left_right_plot(plotvar = 'F0', uniformscale = False, savefigure = False, showfigure = True)

# <codecell>

for plotvar in var_strings:
    make_left_right_plot(plotvar = plotvar, uniformscale = False, savefigure = True, showfigure = False)
    make_left_right_plot(plotvar = plotvar, uniformscale = True, savefigure = True, showfigure = False)

# <codecell>

import subprocess, shlex

# <codecell>

cmd = 'pdftk left-rightRLN.%s.SLN_?.pdf cat output left-rightRLN.%s.SLN_01234.pdf' % ((plotvar, ) * 2)
print shlex.split(cmd)

# <codecell>

subprocess.check_output(cmd, shell = True)

# <codecell>

try:
    subprocess.check_output(shlex.split(cmd), stderr = subprocess.STDOUT)
except subprocess.CalledProcessError as err:
    print err.output
    

# <codecell>

" ".join( sorted(glob.glob('left-rightRLN.%s.SLN_?.pdf' % plotvar)) )

# <codecell>

for plotvar in var_strings:
    cmd = "pdftk left-rightRLN.%s.SLN_?.pdf cat output left-rightRLN.%s.SLN_01234.pdf" % ((plotvar, ) * 2)
    print cmd
    
    r = subprocess.check_output(cmd, shell = True)
    
    cmd = cmd.replace('.pdf', '.normalized.pdf')
    print cmd
    
    r = subprocess.check_output(cmd, shell = True)

# <codecell>

def make_rightRLN_SLN_plot(plotvar = 'F0', uniformscale = True, savefigure = False, showfigure = False, figsize = (15, 12)):
    if figsize == None:
        figsize = mpl.rcParams['figure.figsize']

    colorbarlabel = label_dict[plotvar]

    for leftRLNnum in range(Nlevels):
        fig, ax = plt.subplots(figsize = figsize)
        
        leftRLNcond = 'left RLN %d' % leftRLNnum
        
        ax.set_title(leftRLNcond)
        
        axim = ax.imshow(var_plot[plotvar][leftRLNnum, :, :])
        
        # scale all axis with respect to global min and max
        # uniform_scale = False
        if uniformscale:
            axim.set_clim(np.nanmin(var_plot[plotvar]), np.nanmax(var_plot[plotvar]))
            ax.relim()
        
        co = ax.contour(axim.get_array(), 
                        6, 
                        colors = 'w')
        ax.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)
        
        ax.axis([-0.5, 4.5, -0.5, 7.5])
        
        ax.set_ylabel('right RLN')
        
        ax.grid(False)
        
        if True or leftRLNnum == 0:
            cb = fig.colorbar(axim, ax = ax)
            cb.set_label(colorbarlabel)
        
        if True or leftRLNnum == Nlevels - 1:
            ax.set_xlabel('SLN')
        
        figname = "rightRLN-SLN.%s.%s.pdf" % ( plotvar, leftRLNcond.replace(' ', '_') )

        if uniformscale:
            figname = figname.replace('.pdf', '.normalized.pdf')

        if savefigure:
            print "saving: ", figname
        
            fig.savefig(figname, orientation = 'landscape',
                        papertype = 'letter', format = 'pdf',
                        bbox_inches = 'tight', pad_inches = 0.1)
    if showfigure:
        plt.show()
    else:
        plt.close('all')

# <codecell>

for plotvar in var_strings:
    make_rightRLN_SLN_plot(plotvar = plotvar, uniformscale = False, savefigure = True, showfigure = False)
    make_rightRLN_SLN_plot(plotvar = plotvar, uniformscale = True, savefigure = True, showfigure = False)

# <codecell>

for plotvar in var_strings:
    cmd = "pdftk rightRLN-SLN.%s.left_RLN_?.pdf cat output rightRLN-SLN.%s.left_RLN_01234567.pdf" % ((plotvar, ) * 2)
    print cmd
    
    r = subprocess.check_output(cmd, shell = True)
    
    cmd = cmd.replace('.pdf', '.normalized.pdf')
    print cmd
    
    r = subprocess.check_output(cmd, shell = True)

# <codecell>

mpl.rcParams['figure.figsize'] = (10, 8)

# <codecell>

plt.plot(var_plot['lstrain'].ravel(), var_plot['rstrain'].ravel(), 'r.', zorder = 10)
plt.plot([-15, 35], [-15, 35], 'k', zorder = 1, alpha = 0.5)
plt.axis([-15, 35, -15, 35])

plt.xlabel('left strain [%]')
plt.ylabel('right strain [%]')

plt.show()

# <codecell>

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = 'col', sharey = 'col')

ax1.plot(var_plot['lstrain'].ravel(), var_plot['F0'].ravel(), 'r.', label = 'left strain')
# ax1.xlabel('left strain [%]')
ax1.set_ylabel('F0 [Hz]')
# ax1.xlim(xmin = -20, xmax = 40)

ax1.legend(fontsize = 'small', numpoints = 1, loc = 'upper left')

ax2.plot(var_plot['rstrain'].ravel(), var_plot['F0'].ravel(), 'r.', label = 'right strain')

ax2.set_xlabel('strain [%]')
ax2.set_ylabel('F0 [Hz]')
ax2.legend(fontsize = 'small', numpoints = 1, loc = 'upper left')

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (10, 8))

for SLNnum, SLNcond in enumerate(SLNconditions):
    
    plt.plot(var_plot['lstrain'][:, :, SLNnum].ravel(), var_plot['F0'][:, :, SLNnum].ravel(), 
             symblist[SLNnum], mec = colorlist[SLNnum], mfc = 'None', ms = 10, label = SLNcond, alpha = 0.6)

plt.plot(var_np_plot['lstrain'].ravel(), np.ones_like(var_np_plot['lstrain']).ravel() * 50, 
         '|', mec = 'k', mfc = 'None', ms = 10, label = 'no onset')
    
plt.xlabel('left strain [%]')
plt.ylabel('F0 [Hz]')

plt.xlim(xmin = -20, xmax = 40)
plt.ylim(ymin = 40, ymax = 350)

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), numpoints = 1)

if True:
    figname = "left-rightRLN_F0-strain.pdf"
    fig.savefig(figname, orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (10, 8))

for SLNnum, SLNcond in enumerate(SLNconditions):
    
    plt.plot(var_plot['rstrain'][:, :, SLNnum].ravel(), var_plot['F0'][:, :, SLNnum].ravel(), 
             symblist[SLNnum], mec = colorlist[SLNnum], mfc = 'None', ms = 5, label = SLNcond)

plt.plot(var_np_plot['rstrain'].ravel(), np.ones_like(var_np_plot['rstrain']).ravel() * 50, 
         '|', mec = 'k', mfc = 'None', ms = 10, label = 'no onset')
    
plt.xlabel('right strain [%]')
plt.ylabel('F0 [Hz]')

plt.xlim(xmin = -20, xmax = 40)
plt.ylim(ymin = 40, ymax = 350)

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), numpoints = 1)

plt.show()

# <codecell>

def scatterplot(x, y, z = None, xlabel = None, ylabel = None, zlabel = None, figsize = (13, 8), savename = None):
    fig, ax = plt.subplots(figsize = figsize)
    
    if z is not None:
        z2 = (z.ravel() - np.nanmin(z.ravel()) + 50)**2
        z2shift = (z2 - np.nanmin(z2))
        size = z2shift / np.nanmax(z2shift) * 975 + 25
        
        color = z.ravel()
    else:
        size = 50
        color = 'red'
    
    a1 = ax.scatter(x.ravel(), y.ravel(), 
                    s = size, 
                    marker = 'o', alpha = 0.6, 
                    c = color, 
                    edgecolors = 'None', cmap = cmap)
    
    # a2 = ax.plot(var_np_plot['lstrain'].ravel(), var_np_plot['dVP'].ravel(), '*', 
    #               ms = 10, mec = 'black', mfc = 'None', label = 'no onset')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if z is not None:
        cb = fig.colorbar(a1)
        
        if zlabel is not None:
            cb.set_label(zlabel)

    ax.set_xlim(xmin = 0.5 * np.nanmin(x.ravel()),
                xmax = 1.1 * np.nanmax(x.ravel()))
    
    ax.xaxis.get_major_formatter().set_powerlimits((0, 4))
    
    if savename is not None:
        figname = savename
        fig.savefig(figname, orientation = 'landscape',
                    papertype = 'letter', format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0.1)
        
    plt.show()

# <codecell>

scatterplot(var_plot['dVP'] * var_plot['F0']**2, var_plot['ps'], z = var_plot['F0'],
            xlabel = 'Dvp * F0^2 [a. u.]', ylabel = 'ps [Pa]', zlabel = 'frequency [Hz]', savename = 'left-rightRLN_scaling.pdf')

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

SLNnum = 2

plt.scatter(var_plot['lstrain'][:, :, :].ravel(), var_plot['dVP'][:, :, :].ravel(), 
            s = var_plot['F0'][:, :, :].ravel()**2 * 0.01, 
            marker = 'o', alpha = 0.6,
            c = var_plot['F0'][:, :, :].ravel(), 
            edgecolors = 'None', cmap = cmap)

plt.xlabel('left strain [%]')
plt.ylabel('dVP [%]')

plt.plot(var_np_plot['lstrain'].ravel(), var_np_plot['dVP'].ravel(), '*', ms = 10, mec = 'black', mfc = 'None', label = 'no onset')

cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.xlim(xmin = -20, xmax = 40)

plt.show()

# <codecell>

scatterplot(var_plot['Q'], var_plot['ps'], z = var_plot['F0'], 
            xlabel = 'Q [ml/s]', ylabel = 'ps [Pa]', zlabel = 'frequency [Hz]')

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

plt.scatter(var_plot['Q'].ravel(), var_plot['ps'].ravel(),
            s = var_plot['F0'].ravel()**2 * 0.01, 
            marker = 'o', alpha = 0.6,
            c = var_plot['F0'].ravel(), edgecolors = 'None', cmap = cmap)

plt.plot(var_np_plot['Q'].ravel(), var_np_plot['ps'].ravel(), '*', ms = 10, mec = 'black', mfc = 'None', label = 'no onset')

plt.xlabel('Q [ml/s]')
plt.ylabel('ps [Pa]')

cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.ylim(ymin = 150, ymax = 1100)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

for SLNnum, SLNcond in enumerate(SLNconditions):
    
    plt.plot(var_plot['ps'][:, :, SLNnum].ravel(), var_plot['F0'][:, :, SLNnum].ravel(), 
             symblist[SLNnum], mec = colorlist[SLNnum], mfc = 'None', ms = 5, label = SLNcond)

plt.plot(var_np_plot['ps'].ravel(), np.ones_like(var_np_plot['ps']).ravel() * 25, 
         '|', mec = 'k', mfc = 'None', ms = 10, label = 'no onset')
    
plt.xlabel('onset pressure [Pa]')
plt.ylabel('F0 [Hz]')

plt.xlim(xmin = 150, xmax = 1100)
# plt.ylim(ymin = 45, ymax = 350)

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), numpoints = 1)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

for SLNnum, SLNcond in enumerate(SLNconditions):
    
    plt.plot(var_plot['Q'][:, :, SLNnum].ravel(), var_plot['F0'][:, :, SLNnum].ravel(), 
             symblist[SLNnum], mec = colorlist[SLNnum], mfc = 'None', ms = 5, label = SLNcond)

plt.plot(var_np_plot['Q'].ravel(), np.ones_like(var_np_plot['Q']).ravel() * 25, 
         '|', mec = 'k', mfc = 'None', ms = 10, label = 'no onset')
    
plt.xlabel('onset flow rate [ml/s]')
plt.ylabel('F0 [Hz]')

# plt.xlim(xmin = -20, xmax = 35)
# plt.ylim(ymin = 45, ymax = 350)

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), numpoints = 1)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

for SLNnum, SLNcond in enumerate(SLNconditions):
    
    plt.plot(var_plot['dVP'][:, :, SLNnum].ravel(), var_plot['F0'][:, :, SLNnum].ravel(), 
             symblist[SLNnum], mec = colorlist[SLNnum], mfc = 'None', ms = 5, label = SLNcond)

plt.plot(var_np_plot['dVP'].ravel(), np.ones_like(var_np_plot['dVP']).ravel() * 25, 
         '|', mec = 'k', mfc = 'None', ms = 10, label = 'no onset')
    
plt.xlabel('dVP [%]')
plt.ylabel('F0 [Hz]')

plt.xlim(xmin = 10, xmax = 110)
# plt.ylim(ymin = 45, ymax = 350)

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), numpoints = 1)

plt.show()

# <codecell>

plt.plot(var_plot['A'].ravel(), var_plot['F0'].ravel(), 'r.')
plt.xlabel('A [a.u.]')
plt.ylabel('F0 [Hz]')
plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

for SLNnum, SLNcond in enumerate(SLNconditions):
    
    plt.plot(var_plot['A'][:, :, SLNnum].ravel(), var_plot['F0'][:, :, SLNnum].ravel(), 
             symblist[SLNnum], mec = colorlist[SLNnum], mfc = 'None', ms = 5, label = SLNcond)

plt.plot(var_np_plot['A'].ravel(), np.ones_like(var_np_plot['A']).ravel() * 25, 
         '|', mec = 'k', mfc = 'None', ms = 10, label = 'no onset')
    
plt.xlabel('A [a.u]')
plt.ylabel('F0 [Hz]')

plt.xlim(xmin = 10)
# plt.ylim(ymin = 45, ymax = 350)

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), numpoints = 1)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

plt.scatter(var_plot['dVP'].ravel(), var_plot['A'].ravel(), 
            s = var_plot['F0'].ravel()**2 * 0.01, 
            marker = 'o', alpha = 0.6, 
            c = var_plot['F0'].ravel(), edgecolors = 'None', cmap = cmap)

plt.plot(var_np_plot['dVP'].ravel(), var_np_plot['A'].ravel(), '*', ms = 7, mec = 'black', mfc = 'None', label = 'no onset', zorder = 1)

plt.xlabel('dVP [%]')
plt.ylabel('A [a.u.]')

cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.xlim(xmin = 10, xmax = 110)
plt.ylim(ymin = 15, ymax = 110)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

plt.scatter(var_plot['lstrain'].ravel(), var_plot['A'].ravel(), 
            s = var_plot['F0'].ravel()**2 * 0.01, 
            marker = 'o', alpha = 0.6, 
            c = var_plot['F0'].ravel(), edgecolors = 'None', cmap = cmap)

plt.plot(var_np_plot['lstrain'].ravel(), var_np_plot['A'].ravel(), '*', 
         ms = 7, mec = 'black', mfc = 'None', label = 'no onset', zorder = 1)

plt.xlabel('lstrain [%]')
plt.ylabel('A [a.u.]')

cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.xlim(xmax = 40)
plt.ylim(ymin = 15, ymax = 110)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (13, 8))

for SLNnum, SLNcond in enumerate(SLNconditions):

    plt.scatter(var_plot['lstrain'][:, :, SLNnum].ravel(), var_plot['A'][:, :, SLNnum].ravel(),
                s = var_plot['F0'][:, :, SLNnum].ravel() * 0.4,
                marker = symblist[SLNnum], c = var_plot['F0'][:, :, SLNnum].ravel(), 
                facecolors = 'None', 
                # edgecolors = 'None', 
                cmap = cmap,
                label = SLNcond)

plt.plot(var_np_plot['lstrain'].ravel(), var_np_plot['A'].ravel(), '*', ms = 7, mec = 'black', mfc = 'None', label = 'no onset', zorder = 1)

plt.xlabel('lstrain [%]')
plt.ylabel('A [a.u.]')

cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.xlim(xmax = 40)
plt.ylim(ymin = 15, ymax = 110)

plt.legend(loc = 'upper left', bbox_to_anchor = (1.2, 1), numpoints = 1)

plt.show()

# <codecell>


