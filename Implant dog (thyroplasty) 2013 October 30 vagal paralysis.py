# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

xls_dir = './Implant_2013_10_30/'

# <codecell>

!ls -alot $xls_dir/*.xls*

# <codecell>

basedir = "/extra/InVivoDog/InVivoDog_2013_10_30/data LabView"

# <codecell>

vagal_book = xlrd.open_workbook(filename = os.path.join(xls_dir, 'RSLNvsRRLN.Implants.10.30.13_dc_FINAL.xls'))

# <codecell>

vagal_book.sheet_names()

# <codecell>

# make new dict for data

implant_vagal = {str(item): None for item in vagal_book.sheet_names()}

expname = "right SLN versus right RLN"

# <codecell>

implant_vagal['GOOD_No Implant'] = dict(hdf5datadir = expname + "/baseline no implant")
implant_vagal['GOOD-No Implant Repeat'] = dict(hdf5datadir = expname + "/no implant baseline repeat")

implant_vagal['GOOD Rectangle'] = dict(hdf5datadir = expname + "/rectangular implant")
implant_vagal['GOOD M Divergent'] = dict(hdf5datadir = expname + "/divergent implant")
implant_vagal['GOOD M Convergent'] = dict(hdf5datadir = expname + "/convergent implant")
implant_vagal['GOOD V-Shaped'] = dict(hdf5datadir = expname + "/V-shaped implant")
implant_vagal['GOOD M ELRectangle'] = dict(hdf5datadir = expname + "/long rectangular implant")
implant_vagal['GOOD-ELDivergent'] = dict(hdf5datadir = expname + "/long divergent implant")
implant_vagal['GOOD-ELConvergent'] = dict(hdf5datadir = expname + "/long convergent implant")
implant_vagal['GOOD ELV-shaped'] = dict(hdf5datadir = expname + "/long V-shaped implant")

implant_vagal['GOOD M Baseline'] = dict(hdf5datadir = "SLN versus RLN" + "/baseline")

# <codecell>

for key, value in implant_vagal.items():
    try:
        print "key: {}\tisdir: {}\t#hdf5: {}".format(key, 
               os.path.isdir(os.path.join(basedir, value['hdf5datadir'])), 
               len(glob.glob(os.path.join(basedir, value['hdf5datadir'], '*.hdf5'))))
    except Exception as e:
        print key, value
        print e

# <rawcell>

# !ls -alo "../InVivoDog_2013_10_23/data LabView/right SLN versus right RLN"

# <codecell>

os.listdir("../InVivoDog_2013_10_30/data LabView/" + expname, )

# <codecell>

for casename in vagal_book.sheet_names():
    sheet = vagal_book.sheet_by_name(casename)
    
    print "#: {}, name: {}\t #rows: {}, #cols: {}\t{}".format(sheet.number, 
                                                             sheet.name, 
                                                             sheet.nrows, 
                                                             sheet.ncols, 
                                                             implant_vagal[casename])

# <codecell>

# number of stimulations: (7 + 1) * (7 + 1) = 8 * 8

Nstimulation = (7 + 1)**2
print "Nstimulation: ", Nstimulation

# <codecell>

implant_vagal

# <markdowncell>

# fill data dict implant_vagal with manually measured data from xls(x) files
# ---

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
            if onset.strip().lower() in ['np', '']:
                onset_list.append(np.nan)
        except:
            onset_list.append(int(onset))
            
        try:
            if T.strip().lower() in ['', 'np']:
                T_list.append(np.nan)
        except:
            T_list.append(int(T))
            
    onsettime_ms = np.array(onset_list) / 50.
    F0 = 50.0 * 4.0 / np.array(T_list) * 1000
    
    implant_vagal[casename].update(onsettime_ms = onsettime_ms, F0 = F0)

# <codecell>

implant_vagal

# <codecell>

min_F0 = np.min([np.nanmin(implant_vagal[casename]['F0']) for casename in implant_vagal])
max_F0 = np.max([np.nanmax(implant_vagal[casename]['F0']) for casename in implant_vagal])

print "min_F0: {} Hz, max_F0: {} Hz".format(min_F0, max_F0)

# <codecell>

# need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
    
Voffset_ps = 0.0 # -65.6e-3 # -38.0e-3 # depends on the date
isoampgain_ps = 2.0 # I measured it more accurately, it is slightly more than 2
    
Voffset_Q = -8.85e-3 # 27.8e-3 # 33.0e-3 # depends on the date and on the time of the experiment!!!
isoampgain_Q = 2.0

# <codecell>

%run -i 'tools for implant analysis.py'

# <codecell>

# left vagal nerve paralysis
#
# relative levels from column 1 for right SLN
# and column 3 for right RLN

num_rightSLN = 1
num_rightRLN = 3

relative_nerve_levels = [('rightSLN', num_rightSLN), 
                         ('rightRLN', num_rightRLN)]

min_ps, min_Q = getonsetdata(basedir, implant_vagal, relative_nerve_levels)

# <codecell>

implant_vagal

# <codecell>

# del implant_vagal["GOOD M Baseline"]

# <codecell>

implant_vagal.keys()

# <codecell>

print "before correction"
print "min_ps: ", min_ps
print "min_Q: ", min_Q

# <codecell>

casename = 'GOOD-No Implant Repeat'
casename = 'GOOD M Convergent'
casename = 'GOOD M Baseline'

hdf5dirname = os.path.join(basedir, implant_vagal[casename]['hdf5datadir'])
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
print "min_Q: {}, max_Q: {}".format(np.min(d.allQ), np.max(d.allQ))
print "min_ps: {}, max_ps: {}".format(np.min(d.allps), np.max(d.allps))

print 

print "total offset Q: {} Volts".format( np.min(d.allQ) / d.convQ * Vconv )

print "total offset ps: {} Volts".format( np.min(d.allps) / d.convps * Vconv )

# <codecell>

print Voffset_Q
print Voffset_ps

# <codecell>

np.unravel_index(np.argmin(d.allps), d.allps.shape)

# <codecell>

plt.close('all')
plt.plot(d.allps[: , 1:10]) # * 1 / d.convps * Vconv)

# <codecell>

min_ps_onset = np.min([np.nanmin(implant_vagal[casename]['ps_onset']) for casename in implant_vagal])
max_ps_onset = np.max([np.nanmax(implant_vagal[casename]['ps_onset']) for casename in implant_vagal])

print min_ps_onset, max_ps_onset

# <codecell>

min_Q_onset = np.min([np.nanmin(implant_vagal[casename]['Q_onset']) for casename in implant_vagal])
max_Q_onset = np.max([np.nanmax(implant_vagal[casename]['Q_onset']) for casename in implant_vagal])

print min_Q_onset, max_Q_onset

# <codecell>

# make spectrogram arrays with the measured onset time and onset frequency indicated by horizontal and vertical lines

try:
    del d.allspecs
    del d
except:
    pass

for casename in implant_vagal:
    hdf5dirname = os.path.join(basedir, implant_vagal[casename]['hdf5datadir'])
    if not os.path.isdir(hdf5dirname):
        print "hdf5 directory does not exist: ", hdf5dirname
        continue

    hdf5filename = glob.glob(os.path.join(hdf5dirname, '*.hdf5'))
    if len(hdf5filename) > 1:
        print "found more than one hdf5 data file, take the latest one"
        print
        hdf5filename = sorted(hdf5filename)[-1]
    else:
        hdf5filename = hdf5filename[-1]
        
    print casename
    
    d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

    d.get_all_data()
    
    isoamp_adjustment(d)
    
    hdf5file = os.path.basename(hdf5filename)
    
    d.minps = d.allps.min()
    d.maxps = d.allps.max()
    d.minQ = d.allQ.min()
    d.maxQ = d.allQ.max()

    if hdf5file.startswith('SLN versus RLN No implant') or hdf5file.startswith('SLN versus RLN'):
        grid_xaxis = dict(label = 'RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')

    if hdf5file.startswith('SLN versus RLN, no left RLN') or hdf5file.startswith('SLN versus right RLN'):
        grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')
    
    if hdf5file.startswith('right SLN versus right RLN'):
        grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'right SLN', level = 'rightSLN')

    gridx = implant_vagal[casename]['rightRLN'].ravel()
    gridy = implant_vagal[casename]['rightSLN'].ravel()
    stimind = implant_vagal[casename]['stimind'].ravel()

    otime = implant_vagal[casename]['onsettime_ms'].ravel() / 1000.
    F0 = implant_vagal[casename]['F0'].ravel()
    
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

implant_vagal[casename].keys()

# <codecell>

plotonsetdata(implant_vagal, name_paralysis = 'vagal nerve paralysis', 
              ps_normalized = True, Q_normalized = True)

# <codecell>

print hdf5dirname

d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

d.get_all_data()

# <codecell>

plt.close('all')

xind = 7
yind = 7

otime = implant_vagal[casename]['onsettime_ms'][yind, xind] / 1000.
periodT = 1/implant_vagal[casename]['F0'][yind, xind]
stimnum = implant_vagal[casename]['stimind'][yind, xind]

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

plt.close('all')

try:
    del d.allspecs
    del d
except:
    pass

for casename in implant_vagal:
    hdf5dirname = os.path.join(basedir, implant_vagal[casename]['hdf5datadir'])
    if not os.path.isdir(hdf5dirname):
        continue

    hdf5filename = glob.glob(os.path.join(hdf5dirname, '*.hdf5'))
    if len(hdf5filename) > 1:
        print "found more than one hdf5 data file, take the latest one"
        hdf5filename = sorted(hdf5filename)[-1]
    else:
        hdf5filename = hdf5filename[-1]
        
    print casename
    
    d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))

    d.get_all_data()
    
    isoamp_adjustment(d)

    hdf5file = os.path.basename(hdf5filename)
    
    d.minps = d.allps.min()
    d.maxps = d.allps.max()
    d.minQ = d.allQ.min()
    d.maxQ = d.allQ.max()

    if hdf5file.startswith('SLN versus RLN No implant') or hdf5file.startswith('SLN versus RLN'):
        grid_xaxis = dict(label = 'RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')

    if hdf5file.startswith('SLN versus RLN, no left RLN') or hdf5file.startswith('SLN versus right RLN'):
        grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')
    
    if hdf5file.startswith('right SLN versus right RLN'):
        grid_xaxis = dict(label = 'right RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'right SLN', level = 'rightSLN')

    gridx = implant_vagal[casename]['rightRLN'].ravel()
    gridy = implant_vagal[casename]['rightSLN'].ravel()
    stimind = implant_vagal[casename]['stimind'].ravel()

    otime = implant_vagal[casename]['onsettime_ms'].ravel() / 1000.
    periodT = 1 / implant_vagal[casename]['F0'].ravel()
        
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

from collections import OrderedDict

# <codecell>

nerve_conditions = ["GOOD M Baseline", "GOOD_No Implant", "GOOD-No Implant Repeat",
                    "GOOD Rectangle", "GOOD M Convergent", "GOOD M Divergent", "GOOD V-Shaped",
                    "GOOD M ELRectangle", "GOOD-ELConvergent", "GOOD-ELDivergent", "GOOD ELV-shaped"]

# <codecell>

videosTAconditions = [["SLN versus RLN", "baseline02"],
                                  "baseline no implant", "no implant baseline repeat", 
                                  "rectangular implant", "convergent implant",
                                  "divergent implant", "V-shaped implant",
                                  "long rectangular implant", "long convergent implant",
                                  "long divergent implant", "long V-shaped implant"]

# <codecell>

dd = OrderedDict(zip(nerve_conditions, videosTAconditions))

# <codecell>

dd.keys

# <codecell>

dd.items

# <codecell>

%run -i 'tools for implant analysis.py'

# <codecell>

exportdata2csv(implant_vagal, filename = 'vagal_paralysis_2013_10_30')

# <codecell>

implant_vagal.keys

# <codecell>

casenames = ['GOOD M Baseline', 'GOOD_No Implant', 'GOOD-No Implant Repeat',
             'GOOD Rectangle', 'GOOD M Convergent', 'GOOD M Divergent', 'GOOD V-Shaped',
             'GOOD M ELRectangle', 'GOOD-ELConvergent', 'GOOD-ELDivergent', 'GOOD ELV-shaped']

# <codecell>

casenames

# <codecell>

set.symmetric_difference(set(casenames), set(implant_vagal.keys()))

# <codecell>

Bernoulli_Power(implant_vagal)

# <codecell>

for varname, label in zip(varnames, varlabels):
    plot_boxplot(implant_vagal, casenames, varname = varname, label = label, 
                 title = 'vagal nerve paralysis')

# <codecell>

plt.show

# <codecell>

scatterplot(implant_vagal, casenames, title = 'vagal')

# <codecell>

plt.show()

# <codecell>

import pickle

with open('vagal_paralysis_2013_10_30.pkl', 'wb') as f:
    pickle.dump(implant_vagal, f, protocol = -1)

# <codecell>


