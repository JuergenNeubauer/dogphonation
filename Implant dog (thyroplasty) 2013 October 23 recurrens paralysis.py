# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# <style>
# div.cell, div.text_cell_render{
#   max-width:950px;
#   margin-left:auto;
#   margin-right:auto;
# }
# 
# .rendered_html
# {
#   font-size: 140%;
#   }
# 
# .rendered_html li
# {
#   line-height: 1.8;
#   }
# 
# .rendered_html h1, h2 {
#   text-align:center;
#   font-familly:"Charis SIL", serif;
# }
# </style>

# <codecell>

%matplotlib inline

%config InlineBackend
%config InlineBackend.close_figures = False

import matplotlib as mpl
import matplotlib.pyplot as plt

print mpl.is_interactive()
print plt.isinteractive()

# <codecell>

import sys, os, xlrd, glob
import numpy as np

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')

# <codecell>

import dogdata

# <codecell>

xls_dir = './Implant_2013_10_23/'

# <codecell>

!ls -alot $xls_dir/*.xls*

# <codecell>

recurrens_book = xlrd.open_workbook(filename = os.path.join(xls_dir, '10.23.SLNvsRLN.ImplantsRepeated.xls'))

# oldbook = xlrd.open_workbook(filename = os.path.join(xls_dir, 'REVISED.SLNvsRLN.Implants.xlsx'))

# <codecell>

recurrens_book.sheet_names()

# <codecell>

implant_recurrens = {str(item): None for item in recurrens_book.sheet_names()}

basedir = "/extra/InVivoDog/InVivoDog_2013_10_23/data LabView"

implant_recurrens['Baseline'] = dict(hdf5datadir = "SLN versus RLN/No implant on left side")
implant_recurrens['No Implant (No L RLN)'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/No implant on left side")

implant_recurrens['Rectangle'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/Rectangular implant on left side")
implant_recurrens['Divergent'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/Divergent implant on left side")
implant_recurrens['Convergent'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/Convergent implant on left side")
implant_recurrens['V-Shaped'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/V-shaped implant on left side")

implant_recurrens['ELDivergent'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/Long divergent implant on left side")
implant_recurrens['ELRectangle'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/Long rectangular implant on left side")
implant_recurrens['ELConvergent'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/Long convergent implant on left side")
implant_recurrens['ELV-shaped'] = dict(hdf5datadir = "SLN versus RLN, no left RLN/Long V-shaped implant left")

# <codecell>

for key, value in implant_recurrens.items():
    print (key, os.path.isdir(os.path.join(basedir, value['hdf5datadir'])), 
           len(glob.glob(os.path.join(basedir, value['hdf5datadir'], '*.hdf5'))))

# <codecell>

!ls -alo "../InVivoDog_2013_10_23/data LabView/SLN versus RLN, no left RLN"

# <codecell>

for casename in recurrens_book.sheet_names():
    sheet = recurrens_book.sheet_by_name(casename)
    print sheet.number, sheet.name, sheet.nrows, sheet.ncols, implant_recurrens[casename]

# <codecell>

Nstimulation = 64 # 8 * 8 = (7 + 1) * (7 + 1)

# <codecell>

for casename in recurrens_book.sheet_names():
    sheet = recurrens_book.sheet_by_name(casename)
    
    onset_list = [] # onset time in samples (sampling rate: 50 KHz)
    T_list = []     # period of four cycles in samples
    
    for rownum in range(Nstimulation):
        # start in row 4 to discard the header information (rows 0 to 3)
        onset, dummy, Npeaks, T = sheet.row_values(rownum + 4, start_colx = 1, end_colx = 5)
        
        if Npeaks != 4:
            raise ValueError("Npeaks is NOT 4")
        
        try:
            if onset.strip().lower() in ['np', '']:
                onset_list.append(np.nan)
        except:
            onset_list.append(int(onset))
            
        try:
            if T.strip().lower() in ['', 'np']:
                T_list.append(np.nan)
        except:
            T_list.append(int(T))
            
    # onset_list: time in samples, sampling rate for audio: 50 KHz
    onsettime_ms = np.array(onset_list) / 50.
    
    # onset frequency from period of FOUR (4) consecutive periods, in samples, sampling rate: 50 KHz
    F0 = 50.0 * 4.0 / np.array(T_list) * 1000
    
    implant_recurrens[casename].update(onsettime_ms = onsettime_ms, F0 = F0)

# <codecell>

min_F0 = np.min([np.nanmin(implant_recurrens[casename]['F0']) for casename in implant_recurrens])
max_F0 = np.max([np.nanmax(implant_recurrens[casename]['F0']) for casename in implant_recurrens])

print min_F0, max_F0

# <codecell>

# need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
    
Voffset_ps = -32.3e-3 # -38.0e-3 # depends on the date
isoampgain_ps = 2.0 # I measured it more accurately, it is slightly more than 2
    
Voffset_Q = 23.8e-3 # 33.0e-3 # depends on the date
isoampgain_Q = 2.0

# <codecell>

%run -i 'tools for implant analysis.py'

# <codecell>

# left recurrent nerve paralysis
#
# relative levels from column 0 for left SLN (equal to right SLN)
# and column 3 for right RLN

num_leftSLN = 0
num_rightRLN = 3

relative_nerve_levels = [('SLN', num_leftSLN), 
                         ('rightRLN', num_rightRLN)]

min_ps, min_Q = getonsetdata(basedir, implant_recurrens, relative_nerve_levels)

# <codecell>

implant_recurrens

# <codecell>

print "minima before correction"
print "min_ps: ", min_ps
print "min_Q: ", min_Q

# <codecell>

casename = 'No Implant (No L RLN)'
casename = 'Baseline'

hdf5dirname = os.path.join(basedir, implant_recurrens[casename]['hdf5datadir'])
hdf5filename = glob.glob(os.path.join(hdf5dirname, '*.hdf5'))

# take the latest if many
hdf5filename = sorted(hdf5filename)[-1]
    
print casename
print os.path.basename(hdf5filename)

d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))
d.get_all_data()

Vconv = d.convEMG

# <codecell>

# the maximum flow rate should be around 1700 ml/s, otherwise the iso-stim gain is wrong
print casename

print 'before correction'
print "min_Q: {}, max_Q: {}".format(np.min(d.allQ), np.max(d.allQ))
print "min_ps: {}, max_ps: {}".format(np.min(d.allps), np.max(d.allps))

print 

print "total offset Q: {} Volts".format( np.min(d.allQ) / d.convQ * Vconv )

print "total offset ps: {} Volts".format( np.min(d.allps) / d.convps * Vconv )

# <codecell>

np.unravel_index(np.argmin(d.allps), d.allps.shape)

# <codecell>

plt.close('all')
plt.plot(d.allps[: , 1:10] * 1) # / d.convps * Vconv)

# <codecell>

plt.close('all')
plt.imshow(d.allps, aspect = 4)

plt.colorbar(orientation = 'horizontal')

# <codecell>

min_ps_onset = np.min([np.nanmin(implant_recurrens[casename]['ps_onset']) for casename in implant_recurrens])
max_ps_onset = np.max([np.nanmax(implant_recurrens[casename]['ps_onset']) for casename in implant_recurrens])

print min_ps_onset, max_ps_onset

# <codecell>

min_Q_onset = np.min([np.nanmin(implant_recurrens[casename]['Q_onset']) for casename in implant_recurrens])
max_Q_onset = np.max([np.nanmax(implant_recurrens[casename]['Q_onset']) for casename in implant_recurrens])

print min_Q_onset, max_Q_onset

# <codecell>

# make spectrogram arrays with the measured onset time and onset frequency indicated by horizontal and vertical lines

# num_leftSLN = 0
# num_rightRLN = 3

try:
    del d.allspecs
    del d
except:
    pass

for casename in implant_recurrens:
    hdf5dirname = os.path.join(basedir, implant_recurrens[casename]['hdf5datadir'])
    if not os.path.isdir(hdf5dirname):
        print "hdf5 directory does not exist: ", hdf5dirname
        continue

    hdf5filename = glob.glob(os.path.join(hdf5dirname, '*.hdf5'))
    if len(hdf5filename) > 1:
        print "more than one hdf5 file"
        print hdf5filename
        print "use latest"
        hdf5filename = sorted(hdf5filename)[-1]
        print hdf5filename
    else:
        hdf5filename = hdf5filename[0]
        
    print casename
    
    d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

    d.get_all_data()
    
    isoamp_adjustment(d)

    hdf5file = os.path.basename(hdf5filename)
    
    d.minps = d.allps.min()
    d.maxps = d.allps.max()
    d.minQ = d.allQ.min()
    d.maxQ = d.allQ.max()

    if hdf5file.startswith('SLN versus RLN No implant'):
        grid_xaxis = dict(label = 'RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')

    if hdf5file.startswith('SLN versus RLN, no left RLN'):
        grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')
    
    if hdf5file.startswith('right SLN versus right RLN'):
        grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'right SLN', level = 'rightSLN')

    gridx = implant_recurrens[casename]['rightRLN'].ravel()
    gridy = implant_recurrens[casename]['leftSLN'].ravel()
    stimind = implant_recurrens[casename]['stimind'].ravel()

    otime = implant_recurrens[casename]['onsettime_ms'].ravel() / 1000.
    F0 = implant_recurrens[casename]['F0'].ravel()
    
    for signal in ['psub', 'pout']:
        d.show_spectrograms(signal = signal, fmax = 500,
                            nerve_xaxis = grid_xaxis, nerve_yaxis = grid_yaxis,
                            figsize = (2*24/3, 2*18/3))

        for gind, (x, y) in enumerate(zip(gridx, gridy)):
            # print gind, x, y, d.allspecs.grid[x, y].xind, d.allspecs.grid[x, y].yind, otime[gind] * 1000 * 50
            
            d.allspecs.grid[x, y].axvline(x = otime[gind], lw = 1.5, ls = '--', color = 'black')
            d.allspecs.grid[x, y].axhline(y = F0[gind], lw = 1.5, ls = '--', color = 'black')
        
            d.allspecs.grid[x, y].grid(False)
        
        d.savefigure(label = 'Repeat_Check_Onset', format = 'png')
            
        del d.allspecs

    # break
    del d

# <codecell>

help dogdata.DogData.show_spectrograms

# <codecell>

implant_recurrens[casename].keys()

# <codecell>

plotonsetdata(implant_recurrens, name_paralysis = 'recurrent nerve paralysis', 
              ps_normalized = True, Q_normalized = True)

# <codecell>

d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

d.get_all_data()

# <codecell>

xind = 7
yind = 7

otime = implant_recurrens[casename]['onsettime_ms'][yind, xind] / 1000.
periodT = 1/implant_recurrens[casename]['F0'][yind, xind]
stimnum = implant_recurrens[casename]['stimind'][yind, xind]

print stimnum

try:
    plt.clf()
except:
    pass
plt.plot(d.time_ac, d.allpsub[stimnum, :], lw = 2.0, marker = None)

plt.axvspan(xmin = otime, xmax = otime + 4 * periodT, alpha = 0.5, facecolor = 'green')

plt.xlim(xmin = otime - 15 * periodT, xmax = otime + 15 * periodT)

a = plt.gca()

plt.autoscale(enable = True, axis = 'y', tight = True)

plt.xlabel('time [s]')
plt.title(casename)

# <codecell>

import subplotgrid

# <codecell>

# make amplitude plot arrays with the measured onset time and onset period indicated with a green range

# num_leftSLN = 0
# num_rightRLN = 3

try:
    del d.allspecs
    del d
except:
    pass

for casename in implant_recurrens:
    hdf5dirname = os.path.join(basedir, implant_recurrens[casename]['hdf5datadir'])
    if not os.path.isdir(hdf5dirname):
        continue

    hdf5filename = glob.glob(os.path.join(hdf5dirname, '*.hdf5'))
    if len(hdf5filename) > 1:
        print "found more than on hdf5 file"
        print hdf5filename
        print "use latest"
        hdf5filename = sorted(hdf5filename)[-1]
        print hdf5filename
    else:
        hdf5filename = hdf5filename[0]
        
    print casename
    
    d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

    d.get_all_data()
    
    isoamp_adjustment(d)

    hdf5file = os.path.basename(hdf5filename)
    
    d.minps = d.allps.min()
    d.maxps = d.allps.max()
    d.minQ = d.allQ.min()
    d.maxQ = d.allQ.max()

    if hdf5file.startswith('SLN versus RLN No implant'):
        grid_xaxis = dict(label = 'RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')

    if hdf5file.startswith('SLN versus RLN, no left RLN'):
        grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')
    
    if hdf5file.startswith('right SLN versus right RLN'):
        grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'right SLN', level = 'rightSLN')

    gridx = implant_recurrens[casename]['rightRLN'].ravel()
    gridy = implant_recurrens[casename]['leftSLN'].ravel()
    stimind = implant_recurrens[casename]['stimind'].ravel()

    otime = implant_recurrens[casename]['onsettime_ms'].ravel() / 1000.
    periodT = 1 / implant_recurrens[casename]['F0'].ravel()
        
    for signal in ['psub', 'pout']:
        allspecs = subplotgrid.Specgrid(nrows = d.Nlevels, ncols = d.Nlevels, 
                                        nerve01name = grid_xaxis['label'], 
                                        nerve02name = grid_yaxis['label'],
                                        figsize = (2 * 24/3, 2*18/3))

        if signal == 'psub':
            s = d.allpsub
        if signal == 'pout':
            s = d.allpout

        for gind, (stind, x, y) in enumerate(zip(stimind, gridx, gridy)):
            plotsignal = s[stind, :]
            
            onsettime = otime[gind]
            period = periodT[gind]
            
            allspecs.grid[x, y].plot(d.time_ac, plotsignal, lw = 1.0, marker = None)
            
            allspecs.grid[x, y].axvspan(xmin = onsettime, xmax = onsettime + 4 * period, 
                                        facecolor = 'green', edgecolor = 'green', alpha = 0.5)
            
            if not np.isnan(onsettime):
                allspecs.grid[x, y].set_xlim(xmin = onsettime - 10 * period, xmax = onsettime + 10 * period)
                
                plottime = (d.time_ac > onsettime - 10 * period) * (d.time_ac < onsettime + 10 * period)
                plotvalues = np.where(plottime, plotsignal, np.nan)
                
                allspecs.grid[x, y].set_ylim(ymin = np.nanmin(plotvalues) * 1.1, 
                                             ymax = np.nanmax(plotvalues) * 1.1)
                
            else:
                allspecs.grid[x, y].set_ylim(ymin = plotsignal.min(), ymax = plotsignal.max())
            
            allspecs.grid[x, y].grid(True)
            
            allspecs.grid[x, y].set_xticklabels([])
            allspecs.grid[x, y].set_yticklabels([])
            
        allspecs.grid[0, 0].set_ylabel('signal amplitude')
        
        allspecs.savefig(savename = hdf5filename.replace('.hdf5', '.Repeat_Check_Onset.%s.png' % signal))
            
        del allspecs

    # del d

# <codecell>

plt.clf()
ind = 20
plt.plot(d.time_ac, s[ind, :])

plt.axvspan(xmin = otime[ind], xmax = otime[ind] + 4 * periodT[ind], facecolor = 'green', edgecolor = 'green', alpha = 0.5)

plottime = (d.time_ac > otime[ind] - 15 * periodT[ind]) * (d.time_ac < otime[ind] + 15 * periodT[ind])
plotvalues = np.where(plottime, s[ind, :], np.nan)
                
plt.ylim(ymin = np.nanmin(plotvalues) * 1.1, 
         ymax = np.nanmax(plotvalues) * 1.1)

plt.xlim(xmin = otime[ind] - 15 * periodT[ind], xmax = otime[ind] + 15 * periodT[ind])

# <codecell>

implant_recurrens

# <codecell>

exportdata2csv(implant_recurrens, filename = 'recurrens_paralysis_2013_10_23')

# <codecell>

casename = 'Baseline'

implant_recurrens[casename].keys()

# <codecell>

casenames = ['Baseline', 'No Implant (No L RLN)',
             'Rectangle', 'Convergent', 'Divergent', 'V-Shaped',
             'ELRectangle', 'ELConvergent', 'ELDivergent', 'ELV-shaped']

# <codecell>

Bernoulli_Power(implant_recurrens)

# <codecell>

for varname in varnames:
    statistics(implant_recurrens, varname = varname)

# <codecell>

%run -i 'tools for implant analysis.py'

# <codecell>

for varname, label in zip(varnames, varlabels):
    plot_boxplot(implant_recurrens, casenames, varname = varname, label = label, 
                 title = 'recurrent nerve paralysis')

# <codecell>

for varname, label in zip(varnames, varlabels):
    plot_statistics(casenames, varname = varname, label = label)

# <codecell>

scatterplot(implant_recurrens, casenames, title = 'recurrens')

# <codecell>

casenames

# <codecell>

import pickle

with open('vagal_paralysis_2013_10_23.pkl', 'rb') as f:
    implant_vagal = pickle.load(f)

# <codecell>

implant_vagal.keys

# <codecell>

implant_vagal['No Implant']['onsettime_ms']

# <codecell>

cases = ['Baseline', 'No Implant (No L RLN)', 'Convergent', 'ELConvergent',
         'Baseline', 'No Implant', 'Convergent', 'ELConvergent']

labels = 2 * ['Baseline', 'No Implant', 'Convergent', 'Long Convergent']

paralysistype = 4 * ['recurrens'] + ['recurrens'] + 3 * ['vagal']

markers = 2 * ['o', '*', '^', '>'] # , '^', '<', '>', 'D', 'p', 'h', '8']
# markers = ['o'] * 12

minF0, maxF0 = np.infty, -np.infty
minps, maxps = np.infty, -np.infty
minQ, maxQ = np.infty, -np.infty

for (casename, paralysis) in zip(cases, paralysistype):
    if 'recurrens' in paralysis:
        implant = implant_recurrens
    if 'vagal' in paralysis:
        implant = implant_vagal
        
    # sometimes phonation was determined after stimulation had stopped
    # this sound is due to switching off the stimulation and the flow ramp
    phonation = implant[casename]['onsettime_ms'] < 1500

    minF0 = min(minF0, np.nanmin(implant[casename]['F0'][phonation]))
    maxF0 = max(maxF0, np.nanmax(implant[casename]['F0'][phonation]))
    minps = min(minps, np.nanmin(implant[casename]['ps_onset'][phonation]))
    maxps = max(maxps, np.nanmax(implant[casename]['ps_onset'][phonation]))
    minQ = min(minQ, np.nanmin(implant[casename]['Q_onset'][phonation]))
    maxQ = max(maxQ, np.nanmax(implant[casename]['Q_onset'][phonation]))

# allF0 = np.array([implant_recurrens[casename]['F0'].ravel() for casename in cases]).ravel()
# allA = np.array([implant_recurrens[casename]['A_onset'].ravel() for casename in cases]).ravel()
# allP = np.array([implant_recurrens[casename]['P_onset'].ravel() for casename in cases]).ravel()

# allps = np.array([implant_recurrens[casename]['ps_onset'].ravel() for casename in cases]).ravel()
# allQ = np.array([implant_recurrens[casename]['Q_onset'].ravel() for casename in cases]).ravel()

plt.close('all')
plt.figure(figsize = (20, 15))

a1 = plt.subplot(2, 1, 1)
a2 = plt.subplot(2, 1, 2)

# a3 = plt.subplot(2, 2, 3)
# a4 = plt.subplot(2, 2, 4)

for caseind, (casename, paralysis) in enumerate(zip(cases, paralysistype)):
    if 'recurrens' in paralysis:
        implant = implant_recurrens
    if 'vagal' in paralysis:
        implant = implant_vagal
    
    # A = implant[casename]['A_onset'].ravel()
    # P = implant[casename]['P_onset'].ravel()
    
    # sometimes phonation was determined after stimulation had stopped
    # this sound is due to switching off the stimulation and the flow ramp
    phonation = implant[casename]['onsettime_ms'] < 1500

    Q = implant[casename]['Q_onset'][phonation].ravel()
    ps = implant[casename]['ps_onset'][phonation].ravel()
    F0 = implant[casename]['F0'][phonation].ravel()
    
    # plt.plot(implant_recurrens[casename]['ps_onset'].ravel(), 
    #             implant_recurrens[casename]['Q_onset'].ravel(), 'o', 
    #          label = casename)
    
    # plt.plot(implant_recurrens[casename]['F0'].ravel(), A.ravel(), 'o', label = casename, mec = 'None')
    
    if caseind < 4: # 2:
        ax = a1
        axlabel = 'A'
    elif caseind < 8: # 4:
        ax = a2
        axlabel = 'B'
    elif caseind < 6:
        ax = a3
        axlabel = 'C'
    else:
        ax = a4
        axlabel = 'D'
            
    if caseind % 4 == 0:
        color = 'red'
    elif caseind % 4 == 1:
        color = 'green'
    elif caseind % 4 == 2:
        color = 'blue'
    elif caseind % 4 == 3:
        color = 'black'
            
    scatter = ax.scatter(# A, P, 
                ps, Q,
                # s = F0 * 5,
                s = 200,
                c = color, # F0,
                # vmin = minF0, vmax = maxF0,
                edgecolors = 'none',
                # linewidths = linewidths,
                marker = markers[caseind],
                label = labels[caseind],
                alpha = 0.7)
    
    no_nans = ~np.isnan(ps)
    psdata = ps[no_nans]
    Qdata = Q[no_nans]
    
    a = np.vstack([psdata, np.ones_like(psdata)]).T
    m, c = np.linalg.lstsq(a, Qdata)[0]
    
    ax.plot(ps, m * ps + c, '-', color = color, lw = 5, zorder = 1000)
    
    if False:
        if caseind == 7:
            cb = plt.colorbar(scatter, ax = [a1, a2, a3, a4])
            cb.set_label('onset F0 [Hz]')
        
    if caseind in [0, 1, 2, 3]:
        ax.set_xticklabels([])
    if caseind in [2, 3, 6, 7]:
        # ax.set_yticklabels([])
        pass
        
    if caseind in [4, 5, 6, 7]:
        # ax.set_xticks(range(400, 2000, 400))
        pass

    if caseind in [1, 3]:
        ax.set_title('Recurrent nerve paralysis')
    if caseind in [5, 7]:
        ax.set_title('Vagal nerve paralysis')
        
    if caseind in [1, 3, 5, 7]:
        ax.text(-0.1, 1.0, axlabel, transform = ax.transAxes, fontsize = 40, 
                # bbox = dict(facecolor = 'red', alpha = 1)
                )
        
    if caseind in [4, 6]:
        ax.set_xlabel('onset ps [Pa]')
        # plt.xlabel('onset Bernoulli area [a. u.]')
    if caseind in [0, 4]:
        ax.set_ylabel('onset Q [ml/s]')
        # plt.ylabel('onset aerodynamic power [W]')
        
    ax.set_xlim(xmin = 0.95 * minps, xmax = 1.05 * maxps)
    ax.set_ylim(ymin = 0.95 * minQ, ymax = 1.05 * maxQ)

    # plt.gray()

    if caseind in [0, 1, 2, 3]:
        location = 'upper left'
    if caseind in [4, 5, 6, 7]:
        location = 'upper right'
            
    # location = 'best'
            
    ax.legend(loc = location, # (0, -0.15), 
              scatterpoints = 1, numpoints = 1, 
              framealpha = 1.0,
              fontsize = 'medium',
              labelspacing = 0.15 # default: 0.5
               # mode = 'expand', 
               # ncol = 2
               )

figname = 'Implants_Figure2.pdf'
# figname = '{}.power-area-F0.{}.pdf'.format(title, casename)
plt.savefig(figname, orientation = 'landscape', bbox_inches = 'tight')

# <codecell>

# landmarks at rest and onset clicked by Eli
# if onset didn't occur, used the last frame in recording
# see strainanalysis.py
landmarkdir = "/extra/InVivoDog/Elazar/results/10_23_2013_implant_SLNvsRLN/"

print "%s exists? " % landmarkdir, os.path.isdir(landmarkdir)

clickfiles = sorted(glob.glob(os.path.join(landmarkdir, '*.npz')))

print "found N clickfiles: ", len(clickfiles)

# <codecell>

print implant_recurrens.keys()
print 'number of cases: ', len(implant_recurrens.keys())

# <codecell>

len(clickfiles) / Nstimulation

# <codecell>

# from posture_onset.implant_SLNvsRLN_2013_10_23.py

TAconditions = ["No Implant (No L RLN)", "Rectangle", "Convergent", "Divergent", "V-Shaped",
                         "ELRectangle", "ELConvergent", "ELDivergent", "ELV-shaped"]

cinecasename = "InVivoDog_2013_10_23/SLN versus RLN, no left RLN"

videosTAconditions = ["No implant on left side", "Rectangular implant on left side", "Convergent implant on left side",
                                  "Divergent implant on left side", "V-shaped implant on left side",
                                  "Long rectangular implant on left side", "Long convergent implant on left side",
                                  "Long divergent implant on left side", "Long V-shaped implant left"]

datasavedir = "./results/10_23_2013_implant_SLNvsRLN"

# <codecell>

set.symmetric_difference(set(TAconditions), set(implant_recurrens.keys()))

# <codecell>

with open(clickfiles[0], 'rb') as f:
    clickdat = np.load(f)
os.path.basename(clickfiles[0])

# <codecell>

clickdat.items

# <codecell>

%run -i 'tools for implant analysis.py'

# <codecell>

datetimestamp(clickfiles[0], debug = False)

# <codecell>

stimlevelindex = dict()
onsetframenumber = dict()
baseline_pos = dict()
onset_pos = dict()

for casename in implant_recurrens:

    print 'casename: ', casename
    print 'hdf5datadir: ', implant_recurrens[casename]['hdf5datadir']
    casedescription = implant_recurrens[casename]['hdf5datadir'].replace('/', ' ')
    print 'casedescription: ', casedescription
    
    case_clickfiles = [item for item in clickfiles if casedescription in os.path.basename(item)]
    case_clickfiles = sorted(case_clickfiles, key = datetimestamp)
    
    try:
        # testing the missing case
        case_clickfiles.pop(1)
        case_clickfiles.pop(3)
    except:
        pass
    
    if not case_clickfiles:
        print "ERROR:\tno clickfiles found"
        print
        continue
    
    # this number comes from the numbering of the cine files
    # this numbering wraps over at 999, to be followed by 000, so ordering by date is necessary
    clickfilenumber = np.array([int(os.path.basename(item).split('_')[1]) for item in case_clickfiles])
    
    inc = np.diff(clickfilenumber)
    
    if np.all(inc == 1):
        print "OK: complete sequence of clickfiles found"
    if np.any(inc < 0):
        print "CAUTION:\tclickfile sequence wrapped over"
    if np.any(inc > 1):
        print "ALERT:\t\tat least one clickfile is missing"
        missingfileindices = 1 + np.argwhere(inc > 1).squeeze()
        print 'missing at index: ', missingfileindices
        for ind in missingfileindices:
            case_clickfiles.insert(ind, None)
    
    stimlevelindex[casename] = []
    onsetframenumber[casename] = []
    baseline_pos[casename] = []
    onset_pos[casename] = []
    
    list_strains = []
    list_d_rel = []
    list_stimcoord = []
    
    for itemnum, item in enumerate(case_clickfiles):
        if not item:
            stimlevelindex[casename].append(None)
            onsetframenumber[casename].append(None)
            
            baseline_pos[casename].append(None)
            onset_pos[casename].append(None)
            
            continue
            
        with open(item, 'rb') as f:
            clickdat = np.load(f)
            
            stimlevelindex[casename].append(clickdat['stimlevelindex'].tolist())
            onsetframenumber[casename].append(clickdat['onsetframenumber'].tolist())
          
            baseline_pos[casename].append(clickdat['baseline_pos'].tolist())
            onset_pos[casename].append(clickdat['onset_pos'].tolist())
            
    lstrain = np.ones_like(implant_recurrens[casename]['F0']) * np.nan
    rstrain = lstrain.copy()
            
    for baseline, onset, stimindex in zip(baseline_pos[casename], onset_pos[casename], stimlevelindex[casename]):
        l_baseline, d_baseline = distances(baseline)
        l_onset, d_onset = distances(onset)
        
        if l_baseline is not None:
            strains = (l_onset - l_baseline) / l_baseline * 1000.0
            d_rel = d_onset / d_baseline * 100.0
        else:
            strains = None
            d_rel = None
    
        if stimindex is not None:
            stimcoord = np.argwhere(implant_recurrens[casename]['stimind'] == stimindex).squeeze()
            ind0, ind1 = stimcoord
        else:
            stimcoord = None
            ind0, ind1 = None, None
        
        # lstrain[ind0, ind1] = strains
        
        list_stimcoord.append(stimcoord)
        list_strains.append(strains)
        list_d_rel.append(d_rel)
        
    implant_recurrens[casename]['leftstrain'] = None
    implant_recurrens[casename]['rightstrain'] = None
    implant_recurrens[casename]['dVP'] = None
    
    print

# <codecell>

print casename

for i, j, k in zip(list_stimcoord, list_d_rel, list_strains):
    if i is not None:
        temp_stimcoord = i * np.nan
    if j is not None:
        temp_d_rel = j * np.nan
    if k is not None:
        temp_strains = k * np.nan

# <codecell>

for l, temp in [(list_stimcoord, temp_stimcoord), (list_d_rel, temp_d_rel), (list_strains, temp_strains)]:
    while True:
        try:
            none_pos = l.index(None)
        except:
            break
        else:
            l[none_pos] = temp.copy()

# <codecell>

np.array(list_stimcoord)

# <codecell>

implant_recurrens[casename]['stimind']

# <codecell>


