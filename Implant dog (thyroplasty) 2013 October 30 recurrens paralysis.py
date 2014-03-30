# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

xls_dir = './Implant_2013_10_30/'

# <codecell>

!ls -alot $xls_dir/*.xls*

# <codecell>

basedir = "/extra/InVivoDog/InVivoDog_2013_10_30/data LabView"

# <codecell>

recurrens_book = xlrd.open_workbook(filename = os.path.join(xls_dir, 'SLNvsRLN Implants 10 30 13_dc_FINAL.xls'))

# <codecell>

recurrens_book.sheet_names()

# <codecell>

# make new dict for data

implant_recurrens = {str(item): None for item in recurrens_book.sheet_names()}

expname = "SLN versus right RLN"

# <codecell>

implant_recurrens['GOOD No Implant (No L RLN)'] = dict(hdf5datadir = expname + "/No implant")

implant_recurrens['GOOD Rectangle'] = dict(hdf5datadir = expname + "/rectangular implant")
implant_recurrens['GOOD Divergent'] = dict(hdf5datadir = expname + "/divergent implant")
implant_recurrens['GOOD Convergent'] = dict(hdf5datadir = expname + "/convergent implant")
implant_recurrens['GOOD-V-shape DC'] = dict(hdf5datadir = expname + "/V-shaped implant")

implant_recurrens['GOOD ELRectangle'] = dict(hdf5datadir = expname + "/long rectangular implant")
implant_recurrens['GOOD ELDivergent'] = dict(hdf5datadir = expname + "/long divergent implant")
implant_recurrens['GOOD ELConvergent'] = dict(hdf5datadir = expname + "/long convergent implant")
implant_recurrens['GOOD EL-V Shaped'] = dict(hdf5datadir = expname + "/long V-shaped implant")


del implant_recurrens['GOOD V-Shaped'] # use data from GOOD-V-shape DC

# <codecell>

for key, value in implant_recurrens.items():
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

for casename in recurrens_book.sheet_names():
    sheet = recurrens_book.sheet_by_name(casename)
    
    try:
        print "#: {}, name: {}\t #rows: {}, #cols: {}\t{}".format(sheet.number, 
                                                             sheet.name, 
                                                             sheet.nrows, 
                                                             sheet.ncols, 
                                                             implant_recurrens[casename])
    except:
        print "No data file for sheet: ", casename

# <codecell>

# number of stimulations: (7 + 1) * (7 + 1) = 8 * 8

Nstimulation = (7 + 1)**2
print "Nstimulation: ", Nstimulation

# <codecell>

implant_recurrens

# <markdowncell>

# fill data dict implant_recurrens with manually measured data from xls(x) files
# ---

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
    
    try:
        implant_recurrens[casename].update(onsettime_ms = onsettime_ms, F0 = F0)
    except:
        print "key does not exist: ", casename

# <codecell>

implant_recurrens

# <codecell>

min_F0 = np.min([np.nanmin(implant_recurrens[casename]['F0']) for casename in implant_recurrens])
max_F0 = np.max([np.nanmax(implant_recurrens[casename]['F0']) for casename in implant_recurrens])

print "min_F0: {} Hz, max_F0: {} Hz".format(min_F0, max_F0)

# <codecell>

# need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
    
Voffset_ps = 0.0 # -65.6e-3 # -38.0e-3 # depends on the date
isoampgain_ps = 2.0 # I measured it more accurately, it is slightly more than 2
    
Voffset_Q = -7.63e-3 # 27.8e-3 # 33.0e-3 # depends on the date and on the time of the experiment!!!
isoampgain_Q = 2.0

# <codecell>

%run -i 'tools for implant analysis.py'

# <codecell>

# left recurrent nerve paralysis
#
# relative levels from column 1 for right SLN (equals left SLN)
# and column 3 for right RLN

num_rightSLN = 1
num_rightRLN = 3

relative_nerve_levels = [('SLN', num_rightSLN), 
                         ('rightRLN', num_rightRLN)]

min_ps, min_Q = getonsetdata(basedir, implant_recurrens, relative_nerve_levels)

# <codecell>

implant_recurrens.keys()

# <codecell>

print "before correction"
print "min_ps: ", min_ps
print "min_Q: ", min_Q

# <codecell>

casename = 'GOOD No Implant (No L RLN)'
casename = 'GOOD EL-V Shaped'

hdf5dirname = os.path.join(basedir, implant_recurrens[casename]['hdf5datadir'])
hdf5filename = glob.glob(os.path.join(hdf5dirname, '*.hdf5'))
hdf5filename = hdf5filename[0]
    
print casename

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

np.unravel_index(np.argmin(d.allps), d.allps.shape)

# <codecell>

plt.close('all')
plt.plot(d.allps[: , 1:10] * 2) #  / d.convps * Vconv)

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
        continue
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

    if hdf5file.startswith('SLN versus RLN No implant') or hdf5file.startswith('SLN versus RLN'):
        grid_xaxis = dict(label = 'RLN', level = 'rightRLN')
        grid_yaxis = dict(label = 'SLN', level = 'rightSLN')

    if hdf5file.startswith('SLN versus RLN, no left RLN') or hdf5file.startswith('SLN versus right RLN'):
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

print "casename: ", casename
implant_recurrens[casename].keys()

# <codecell>

plotonsetdata(implant_recurrens, name_paralysis = 'recurrent nerve paralysis', 
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

%run -i 'tools for implant analysis.py'

# <codecell>

exportdata2csv(implant_recurrens, filename = 'recurrens_paralysis_2013_10_30')

# <codecell>

implant_recurrens.keys

# <codecell>

casenames = ['No Implant (No L RLN)',
             'Rectangle', 'Convergent', 'Divergent', 'V-shape DC',
             'ELRectangle', 'ELConvergent', 'ELDivergent', 'EL-V Shaped']

casenames = ['GOOD {}'.format(item) for item in casenames]

casenames[4] = casenames[4].replace('GOOD ', 'GOOD-')

# <codecell>

casenames

# <codecell>

Bernoulli_Power(implant_recurrens)

# <codecell>

for varname, label in zip(varnames, varlabels):
    plot_boxplot(implant_recurrens, casenames, varname = varname, label = label, 
                 title = 'recurrent nerve paralysis')

# <codecell>

plt.show

# <codecell>

scatterplot(implant_recurrens, casenames, title = 'recurrens')

# <codecell>

plt.show

# <codecell>

import pickle

with open('vagal_paralysis_2013_10_30.pkl', 'rb') as f:
    implant_vagal = pickle.load(f)

# <codecell>

implant_recurrens.keys()

# <codecell>

implant_vagal.keys()

# <codecell>

phonation = implant_recurrens['GOOD ELDivergent']['onsettime_ms'] < 1500
implant_recurrens['GOOD ELDivergent']['onsettime_ms'][phonation]

# <codecell>

implant_vagal['GOOD-ELConvergent']['onsettime_ms']

# <codecell>

cases = ['GOOD M Baseline', 'GOOD No Implant (No L RLN)', 
         'GOOD ELRectangle', 'GOOD ELConvergent', 'GOOD ELDivergent', 'GOOD EL-V Shaped',
         'GOOD M Baseline', 
         # 'GOOD_No Implant', 
         'GOOD-No Implant Repeat', 
         'GOOD M ELRectangle', 'GOOD-ELConvergent', 'GOOD-ELDivergent', 'GOOD ELV-shaped']

labels = ['Baseline', 'No Implant', 'Long Rectangle', 'Long Convergent', 'Long Divergent', 'Long V-shaped']
labels += ['Baseline', 'No Implant', 
           # 'No Implant Repeat', 
           'Long Rectangle', 'Long Convergent', 'Long Divergent', 'Long V-shaped']

# where does the data come from?
paralysistype = ['vagal'] + 5 * ['recurrens'] + 6 * ['vagal']

markers = ['o', '*', '^', '>', 'v', '<'] + ['o', '*', '^', '>', 'v', '<']
# markers = ['o'] * 12

colors = ['red', 'green', 'blue', 'black', 'cyan', 'orange', 'magenta', 'yellow', 'orange']

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
    
    if caseind < 6:
        ax = a1
        axlabel = 'A'
    elif caseind < 13:
        ax = a2
        axlabel = 'B'
    elif caseind < 6:
        ax = a3
        axlabel = 'C'
    else:
        ax = a4
        axlabel = 'D'

    color = colors[caseind % 6]
    
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
        
    if caseind in range(6):
        ax.set_xticklabels([])
    if caseind in [2, 3, 6, 7]:
        # ax.set_yticklabels([])
        pass
        
    if caseind in [4, 5, 6, 7]:
        # ax.set_xticks(range(400, 2000, 400))
        pass

    if caseind in [0]:
        ax.set_title('Recurrent nerve paralysis')
    if caseind in [6]:
        ax.set_title('Vagal nerve paralysis')
        
    if caseind in [1, 3, 5, 7]:
        ax.text(-0.1, 1.0, axlabel, transform = ax.transAxes, fontsize = 40, 
                # bbox = dict(facecolor = 'red', alpha = 1)
                )
        
    if caseind in [6]:
        ax.set_xlabel('onset ps [Pa]')
        # plt.xlabel('onset Bernoulli area [a. u.]')
    if caseind in [0, 6]:
        ax.set_ylabel('onset Q [ml/s]')
        # plt.ylabel('onset aerodynamic power [W]')
        
    ax.set_xlim(xmin = 0.95 * minps, xmax = 1.05 * maxps)
    ax.set_ylim(ymin = 0.95 * minQ, ymax = 1.05 * maxQ)

    # plt.gray()

    if caseind in [5]:
        location = 'lower right'
    if caseind in [11]:
        location = 'lower right'
        
    if caseind in [5, 11]:
        ax.legend(loc = location, # (0, -0.15), 
                  scatterpoints = 1, numpoints = 1, 
                  framealpha = 0.90,
                  fontsize = 'medium',
                  labelspacing = 0.15 # default: 0.5
                   # mode = 'expand', 
                   # ncol = 2
                   )

figname = 'Implants_Figure3.pdf'
# figname = '{}.power-area-F0.{}.pdf'.format(title, casename)
plt.savefig(figname, orientation = 'landscape', bbox_inches = 'tight')

# <codecell>

plt.show()

# <codecell>


