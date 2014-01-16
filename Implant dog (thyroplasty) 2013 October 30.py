# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os, xlrd, glob
import numpy as np

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')

# <codecell>

import dogdata

# <codecell>

xls_dir = './Implant_2013_10_30/'

# <codecell>

!ls -alot $xls_dir

# <codecell>

recurrens_xls = "DONE SLNvsRLN.Implants.10.30.13.xls"

# <codecell>

recurrens_book = xlrd.open_workbook(filename = os.path.join(xls_dir, recurrens_xls))

# <codecell>

recurrens_book.sheet_names()

# <codecell>

# prepare a dictionary for the data
implant_recurrens = {str(item): None for item in recurrens_book.sheet_names()}

# <codecell>

basedir = "/extra/InVivoDog/InVivoDog_2013_10_30/data LabView"

# <codecell>

ls -alo "$basedir/SLN versus right RLN"

# <codecell>

# implant_recurrens['Baseline'] = dict(hdf5datadir = "SLN versus RLN/No implant on left side")

subdir = "SLN versus right RLN"

implant_recurrens['No Implant (No L RLN)'] = dict(hdf5datadir = os.path.join(subdir, "No implant"))

implant_recurrens['Rectangle'] = dict(hdf5datadir = os.path.join(subdir, "rectangular implant"))
implant_recurrens['Divergent'] = dict(hdf5datadir = os.path.join(subdir, "divergent implant"))
implant_recurrens['Convergent'] = dict(hdf5datadir = os.path.join(subdir, "convergent implant"))
implant_recurrens['V-Shaped'] = dict(hdf5datadir = os.path.join(subdir, "V-shaped implant"))

implant_recurrens['ELDivergent'] = dict(hdf5datadir = os.path.join(subdir, "long divergent implant"))
implant_recurrens['ELRectangle'] = dict(hdf5datadir = os.path.join(subdir, "long rectangular implant"))
implant_recurrens['ELConvergent'] = dict(hdf5datadir = os.path.join(subdir, "long convergent implant"))
implant_recurrens['ELV-shaped'] = dict(hdf5datadir = os.path.join(subdir, "long V-shaped implant"))

# <codecell>

for key, value in implant_recurrens.items():
    print (key, os.path.isdir(os.path.join(basedir, value['hdf5datadir'])), 
           len(glob.glob(os.path.join(basedir, value['hdf5datadir'], '*.hdf5'))))

# <codecell>

!ls -alo "../InVivoDog_2013_10_30/data LabView/SLN versus right RLN"

# <codecell>

ls -aloth "Implant_2013_10_30/"

# <codecell>

vagal_xls = "DONE 10.30.13.RSLNvsRRLN.xls"

# <codecell>

vagal_book = xlrd.open_workbook(filename = os.path.join(xls_dir, vagal_xls))

# <codecell>

vagal_book.sheet_names()

# <codecell>

implant_vagal = {str(item): None for item in vagal_book.sheet_names()}

# <codecell>

subdir = "right SLN versus right RLN"

implant_vagal['No Implant'] = dict(hdf5datadir = "right SLN versus right RLN/No implant")
implant_vagal['Rectangle'] = dict(hdf5datadir = "right SLN versus right RLN/Rectangular implant")
implant_vagal['Divergent'] = dict(hdf5datadir = "right SLN versus right RLN/Divergent implant")
implant_vagal['Convergent'] = dict(hdf5datadir = "right SLN versus right RLN/Convergent implant")
implant_vagal['V-Shaped'] = dict(hdf5datadir = "right SLN versus right RLN/V-shaped implant")
implant_vagal['ELRectangle'] = dict(hdf5datadir = "right SLN versus right RLN/Long rectangular implant")
implant_vagal['ELDivergent'] = dict(hdf5datadir = "right SLN versus right RLN/Long divergent implant")
implant_vagal['ELConvergent'] = dict(hdf5datadir = "right SLN versus right RLN/Long convergent implant")
implant_vagal['ELV-shaped'] = dict(hdf5datadir = "right SLN versus right RLN/Long V-shaped implant")

# <codecell>

for key, value in implant_vagal.items():
    print (key, os.path.isdir(os.path.join(basedir, value['hdf5datadir'])), 
           len(glob.glob(os.path.join(basedir, value['hdf5datadir'], '*.hdf5'))))

# <codecell>

!ls -alo "../InVivoDog_2013_10_23/data LabView/right SLN versus right RLN"

# <codecell>

for casename in recurrens_book.sheet_names():
    sheet = recurrens_book.sheet_by_name(casename)
    print sheet.number, sheet.name, sheet.nrows, sheet.ncols, implant_recurrens[casename]

# <codecell>

for casename in vagal_book.sheet_names():
    sheet = vagal_book.sheet_by_name(casename)
    print sheet.number, sheet.name, sheet.nrows, sheet.ncols, implant_vagal[casename]

# <codecell>

Nstimulation = 64 # 8 * 8 = (7 + 1) * (7 + 1)

# <codecell>

for casename in recurrens_book.sheet_names():
    sheet = recurrens_book.sheet_by_name(casename)
    
    onset_list = [] # onset time in samples (sampling rate: 50 KHz)
    T_list = []     # period of four cycles in samples
    
    for rownum in range(Nstimulation):
        onset, dummy, Npeaks, T = sheet.row_values(rownum + 4, start_colx = 1, end_colx = 5)
        
        if Npeaks != 4:
            raise ValueError("Npeaks is NOT 4")
        
        try:
            if onset.strip() in ['NP', '']:
                onset_list.append(np.nan)
        except:
            onset_list.append(int(onset))
            
        try:
            if T.strip() in ['']:
                T_list.append(np.nan)
        except:
            T_list.append(int(T))
            
    # onset_list: time in samples, sampling rate for audio: 50 KHz
    onsettime_ms = np.array(onset_list) / 50.
    
    # onset frequency from period of FOUR (4) consecutive periods, in samples, sampling rate: 50 KHz
    F0 = 50.0 * 4.0 / np.array(T_list) * 1000
    
    implant_recurrens[casename].update(onsettime_ms = onsettime_ms, F0 = F0)

# <codecell>

for casename in vagal_book.sheet_names():
    sheet = vagal_book.sheet_by_name(casename)
    
    onset_list = [] # onset time in samples (sampling rate: 50 KHz)
    T_list = []     # period of four cycles in samples
    
    for rownum in range(Nstimulation):
        onset, dummy, Npeaks, T = sheet.row_values(rownum + 4, start_colx = 1, end_colx = 5)
        
        if Npeaks != 4:
            raise ValueError("Npeaks is NOT 4")
        
        try:
            if onset.strip() in ['NP', '']:
                onset_list.append(np.nan)
        except:
            onset_list.append(int(onset))
            
        try:
            if T.strip() in ['']:
                T_list.append(np.nan)
        except:
            T_list.append(int(T))
            
    onsettime_ms = np.array(onset_list) / 50.
    F0 = 50.0 * 4.0 / np.array(T_list) * 1000
    
    implant_vagal[casename].update(onsettime_ms = onsettime_ms, F0 = F0)

# <codecell>

%matplotlib inline

# <codecell>

%config InlineBackend

# <codecell>

%config InlineBackend.close_figures = False

# <codecell>

import matplotlib as mpl
import matplotlib.pyplot as plt

# <codecell>

print mpl.is_interactive()
print plt.isinteractive()

# <codecell>

min_F0 = np.min([np.nanmin(implant_recurrens[casename]['F0']) for casename in implant_recurrens])
max_F0 = np.max([np.nanmax(implant_recurrens[casename]['F0']) for casename in implant_recurrens])

print min_F0, max_F0

# <codecell>

# need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
    
Voffset_ps = -38.0e-3 # depends on the date
isoampgain_ps = 2.0 # I measured it more accurately, it is slightly more than 2
    
Voffset_Q = 33.0e-3 # depends on the date
isoampgain_Q = 2.0

# <codecell>

num_leftSLN = 0
num_rightRLN = 3

for casename in implant_recurrens:
    hdf5dirname = os.path.join(basedir, implant_recurrens[casename]['hdf5datadir'])
    if not os.path.isdir(hdf5dirname):
        continue

    hdf5filename = glob.glob(os.path.join(hdf5dirname, '*.hdf5'))
    if len(hdf5filename) > 1:
        continue
    else:
        hdf5filename = hdf5filename[0]
        
    print casename
    
    d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

    d.get_all_data()
    
    # need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
    Vconv = d.convEMG # EMG conversion is just conversion from numbers to Volts
    
    d.allps = isoampgain_ps * d.allps - Voffset_ps / Vconv * d.convps
    
    d.allQ = isoampgain_Q * d.allQ - Voffset_Q / Vconv * d.convQ
    
    F0 = np.ones((d.Nlevels, d.Nlevels)) * np.nan
    onsettime_ms = np.ones_like(F0) * np.nan
    ps_onset = np.ones_like(F0) * np.nan
    leftSLN = np.ones_like(F0) * np.nan
    rightRLN = np.ones_like(F0) * np.nan
    stimind = np.ones_like(F0) * np.nan

    for stimnum, (SLNlevel, rightRLNlevel) in enumerate(d.a_rellevels[:, [num_leftSLN, num_rightRLN]]):
        nerve_xaxis = rightRLNlevel
        nerve_yaxis = SLNlevel
        
        stimind[nerve_yaxis, nerve_xaxis] = stimnum
        leftSLN[nerve_yaxis, nerve_xaxis] = SLNlevel
        rightRLN[nerve_yaxis, nerve_xaxis] = rightRLNlevel
        
        onsettime_ms[nerve_yaxis, nerve_xaxis] = implant_recurrens[casename]['onsettime_ms'][stimnum]
        F0[nerve_yaxis, nerve_xaxis] = implant_recurrens[casename]['F0'][stimnum]
        
        ps_onset[nerve_yaxis, nerve_xaxis] = np.interp(onsettime_ms[nerve_yaxis, nerve_xaxis] / 1000., 
                                                       d.time_psQ, d.allps[stimnum, :],
                                                       left = np.nan, right = np.nan)
    implant_recurrens[casename]['leftSLN'] = leftSLN
    implant_recurrens[casename]['rightRLN'] = rightRLN
    implant_recurrens[casename]['stimind'] = stimind
    
    implant_recurrens[casename]['F0'] = F0
    implant_recurrens[casename]['onsettime_ms'] = onsettime_ms
    implant_recurrens[casename]['ps_onset'] = ps_onset

# <codecell>

# the maximum flow rate should be around 1700 ml/s, otherwise the iso-stim gain is wrong
print casename
print np.min(d.allQ), np.max(d.allQ)

# <codecell>

min_ps_onset = np.min([np.nanmin(implant_recurrens[casename]['ps_onset']) for casename in implant_recurrens])
max_ps_onset = np.max([np.nanmax(implant_recurrens[casename]['ps_onset']) for casename in implant_recurrens])

print min_ps_onset, max_ps_onset

# <codecell>

num_leftSLN = 0
num_rightRLN = 3

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
        continue
    else:
        hdf5filename = hdf5filename[0]
        
    print casename
    
    d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

    d.get_all_data()
    
    # need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
    Vconv = d.convEMG # EMG conversion is just conversion from numbers to Volts
    
    d.allps = isoampgain_ps * d.allps - Voffset_ps / Vconv * d.convps
    
    d.allQ = isoampgain_Q * d.allQ - Voffset_Q / Vconv * d.convQ

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
        d.show_spectrograms(signal = signal, 
                            nerve_xaxis = grid_xaxis, nerve_yaxis = grid_yaxis,
                            figsize = (2*24/3, 2*18/3))

        for gind, (stind, x, y) in enumerate(zip(stimind, gridx, gridy)):
            # print gind, stind+1, x, y, d.allspecs.grid[x, y].xind, d.allspecs.grid[x, y].yind, otime[gind] * 1000 * 50
            
            d.allspecs.grid[x, y].axvline(x = otime[gind], lw = 1.5, ls = '--', color = 'black')
            d.allspecs.grid[x, y].axhline(y = F0[gind], lw = 1.5, ls = '--', color = 'black')
        
            d.allspecs.grid[x, y].grid(False)
        
        d.savefigure(label = 'Check_Onset', format = 'png')
            
        del d.allspecs

    # break
    del d

# <codecell>

for gind, (stind, x, y) in enumerate(zip(stimind, gridx, gridy)):
    print gind, stind+1, x, y, otime[gind] * 1000 * 50

# <codecell>

zip(stimind+1, gridx, gridy, otime * 1000 * 50)

# <codecell>

implant_recurrens[casename].keys()

# <codecell>

for casename in implant_recurrens:
    F0 = implant_recurrens[casename]['F0']
    
    try:
        plt.clf()
    except:
        pass
    
    plt.imshow(F0)
    
    plt.xlabel('right RLN')
    plt.ylabel('SLN')
    
    plt.clim(vmin = min_F0, vmax = max_F0)
    
    plt.title("recurrent nerve paralysis: %s" % casename)
    
    cb = plt.colorbar()
    cb.set_label('frequency [Hz]')
    
    plt.savefig("recurrent_nerve_paralysis.F0.%s.pdf" % casename, 
                orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)
    
    ps_onset = implant_recurrens[casename]['ps_onset']
    
    plt.clf()
    
    plt.imshow(ps_onset)
    
    plt.xlabel('right RLN')
    plt.ylabel('SLN')
    
    # plt.clim(vmin = min_ps_onset, vmax = max_ps_onset)
    
    plt.title("recurrent nerve paralysis: %s" % casename)
    
    cb = plt.colorbar()
    cb.set_label('ps [Pa]')
    
    plt.savefig("recurrent_nerve_paralysis.Ps.%s.pdf" % casename,
                orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)

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

num_leftSLN = 0
num_rightRLN = 3

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
        continue
    else:
        hdf5filename = hdf5filename[0]
        
    print casename
    
    d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

    d.get_all_data()
    
    # need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
    Vconv = d.convEMG # EMG conversion is just conversion from numbers to Volts
    
    d.allps = isoampgain_ps * d.allps - Voffset_ps / Vconv * d.convps
    
    d.allQ = isoampgain_Q * d.allQ - Voffset_Q / Vconv * d.convQ

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
        
        allspecs.savefig(savename = hdf5filename.replace('.hdf5', '.Check_Onset.%s.png' % signal))
            
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


