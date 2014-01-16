# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib as mpl
import matplotlib.pyplot as plt

import os, sys, glob

import numpy as np

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')

# <codecell>

import dogdata
reload(dogdata)

# <codecell>

datadir = "/extra/InVivoDog/InVivoDog_2013_12_18/data LabView/right RLN"

# <codecell>

stimulated_nerve = os.path.split(datadir)[1]
print 'stimulated nerve: ', stimulated_nerve

# <codecell>

ls -alot '/extra/InVivoDog/InVivoDog_2013_12_18/data LabView/'

# <codecell>

!tree "$datadir"

# <codecell>

sub_datadir = os.listdir(datadir)
sub_datadir

# <codecell>

for item in sub_datadir:
    try:
        os.makedirs(os.path.join(datadir, item, "EMG analysis"))
    except:
        pass

# <codecell>

allhdf5files = glob.glob('/extra/InVivoDog/InVivoDog_2013_12_18/data LabView/*/*/*.hdf5')

# <codecell>

import time, datetime

# <codecell>

def datetimestamp(filename):
    
    fname = os.path.basename(filename)

    cleanf = os.path.splitext(fname)[0]

    # all our experiments are on Wednesdays!!!
    datestring = cleanf.split('Wed')[-1].strip()

    return time.strptime(datestring, '%b %d %Y %H %M %S')

# <codecell>

time.localtime() > datetimestamp(allhdf5files[-1])

# <codecell>

hdf5datetime = datetimestamp(allhdf5files[-1])

print hdf5datetime

time.struct_time((2013, 12, 1, 0, 0, 0, 0, 0, 0)) < hdf5datetime

# <codecell>

orderedhdf5files = sorted(allhdf5files, key = datetimestamp)

# <codecell>

dayofexperiment = time.strptime(
                                time.strftime('%d %m %Y', datetimestamp(orderedhdf5files[0])),
                                '%d %m %Y')
print dayofexperiment

def timeduringexperiment(hourminutes = '12:00', dayofexperiment = dayofexperiment):
    return time.strptime( time.strftime('%d %m %Y', dayofexperiment) + ' ' + ' '.join(hourminutes.split(':')), '%d %m %Y %H %M')

timeduringexperiment('14:25')

# <codecell>

signalnames = ['EMG1', 'EMG2', 'EMG3', 'EMG4']
nervenames = ['right medial CT', 'right TA', 'right LCA', 'right PCA']

EMG = dict(zip(signalnames, nervenames))
print EMG

gains = {'EMG' + str(numitem + 1): item for numitem, item in enumerate([1e3, 1e3, 1e3, 1e3])}

def updategains(newgaindict = gains):
    gains.update(newgaindict)
    return [gains[item] for item in sorted(gains)]

def updateEMG(newEMGdict = EMG):
    EMG.update(newEMGdict)
    return [EMG[item] for item in sorted(EMG)]

print updategains()
print updateEMG()

events = [dict(time = '13:30', 
               nervenames = updateEMG(),
               gains = updategains()
               ),
          dict(time = '14:25',
               gains = updategains(zip(signalnames, [1e4, 1e4, 1e4, 1e2]))
               ),
          dict(time = '14:33',
               gains = updategains(zip(signalnames, [1e3, 1e3, 1e3]))
               ),
          dict(time = '14:37',
               gains = updategains(dict(EMG4 = 1000.0))
               ),
          dict(time = '14:42',
               gains = updategains(dict(EMG4 = 100.0))
               ),
          dict(time = '15:10',
               gains = updategains(zip(signalnames[1:], [1e4, 1e4, 1e4]))
               ),
          dict(time = '15:25',
               nervenames = updateEMG(dict(EMG4 = 'right lateral/oblique CT')),
               gains = updategains(dict(EMG4 = 1000.0))
               ),
          dict(time = '16:15',
               nervenames = updateEMG(zip(signalnames, ['left TA', 'left PCA', 'left medial CT', 'left lateral CT'])),
               gains = updategains(zip(signalnames, [1e4, 1e4, 1e3, 1e3]))
               )
          ]

# <codecell>

for event in sorted(events, key = lambda x: x['time']):
    if event.has_key('nervenames'):
        oldnervenames = event['nervenames']
    else:
        event['nervenames'] = oldnervenames
    print "time: {time}\n\tnervenames: {nervenames}\n\tgains: {gains}".format(**event)

# <codecell>

event

# <codecell>

[os.path.basename(item) for item in orderedhdf5files 
 if datetimestamp(item) > timeduringexperiment(events[0]['time'])
 if datetimestamp(item) < timeduringexperiment(events[1]['time'])]

# <codecell>

ls -alot "/extra/InVivoDog/InVivoDog_2013_12_18/data LabView/right SLN"

# <codecell>

ls -alot "$datadir"

# <codecell>

ls -alot "$datadir"/"range finding and EMG"

# <codecell>

ls -alot "$datadir"/"range finding and EMG long run"

# <codecell>

ls -alot "$datadir"/"range finding and EMG 30Hz"

# <codecell>

stimulationcondition = "range finding and EMG 80Hz" # "range finding and EMG long run" # "range finding and EMG 30Hz"

datadirname = os.path.join(datadir, stimulationcondition)
datafilenamestart = "%s %s" % (stimulated_nerve, stimulationcondition)
print 'datadirname: ', datadirname
print 'datafilenamestart: ', datafilenamestart

filenames = [os.path.basename(item) for item in glob.glob(os.path.join(datadirname, datafilenamestart + '*.hdf5'))]
print filenames

d = dogdata.DogData(datadir = datadirname,
                    datafile = filenames[1])

# <codecell>

d.datafilename

# <codecell>

d.timestamp = datetimestamp(d.datafilename)

e = [event for event in events if timeduringexperiment(hourminutes = event['time']) < d.timestamp][-1]

for key in ['gains', 'nervenames']:
    setattr(d, key, e[key])

d.signalnames = signalnames

d.stimulated_nerve = stimulated_nerve

# <codecell>

print d.stimulated_nerve
print d.signalnames
print d.nervenames
print d.gains

# <codecell>

frep_parsed = [float(item.replace('Hz', '')) for item in stimulationcondition.split() if item.endswith('Hz')]

if not frep_parsed:
    print "assume frep = 100Hz"
    frep = 100
else:
    frep = frep_parsed[0]

print "frep = ", frep

d.frep = frep

# <rawcell>

# frep = 100 # pulse repetition rate
# frep = 60
# frep = 30
# 
# frep = 100
# frep = 80

# <codecell>

d.get_all_data()

# <codecell>

NEMG, Nstimulations, Nsamples = d.allEMG.shape

print d.allEMG.shape
print d.NEMG
print d.Nlevels
print d.Nnerves
print d.Nrecnums

# <codecell>

stimulationtime = 1.5 # stimulation pulse train in seconds, typically 1.5 sec
Nend = d.fs_EMG * stimulationtime
print Nend

# <codecell>

# signalnames = ['EMG1', 'EMG2', 'EMG3', 'EMG4']
# nervenames = ['right medial CT', 'right TA', 'right LCA', 'right PCA']

# <codecell>

nervenum = 2

signalname = d.signalnames[nervenum]
nervename = d.nervenames[nervenum]
gain = d.gains[nervenum]

print 'nervenum: ', nervenum
print 'signal: ', signalname
print 'nervename: ', nervename
print 'gain: ', gain

lowfps = frep / 3.0
print 'lowfps = ', lowfps

for stimrep in range(Nstimulations):
    rav, ravtime_ms = d.running_average(signal = signalname, stimulationindex = stimrep + 1, lowfps = lowfps, power = 1, 
                                        window = 'kaiser')
    rav2, rav2time_ms = d.running_average(signal = signalname, stimulationindex = stimrep + 1, lowfps = lowfps, power = 2, 
                                          window = 'kaiser')
    
    if stimrep == 0:
        rastd = np.empty((Nstimulations, len(rav)), dtype = rav.dtype)
        
    rastd[stimrep, :] = np.sqrt(rav2 - rav**2) / gain
    
print rastd.shape

# <codecell>

plt.imshow(rastd * 1000, aspect = 'auto')

cb = plt.colorbar()
cb.set_label('EMG potential [mV]')
plt.title('running std: %s: %s' % (signalname, nervename))

plt.ylabel('stimulation index')

plt.xticks(np.arange(0, 50000, 12500 / 2), np.arange(0, 50000, 12500 / 2) / 25)
plt.xlabel('time [ms]')

plt.yticks(np.arange(0, d.Nrecnums, 4))

plt.xlim(xmax = stimulationtime * d.fs_EMG)

# <codecell>

d.running_average_info

# <codecell>

plt.plot(d.time_EMG * 1000, d.allEMG[nervenum, stimrep, :] / gain * 1000)
# plt.plot(ravtime_ms, rav)
# plt.plot(rav2time_ms, rav2)

plt.plot(rav2time_ms, rastd[stimrep, :] * 1000)

plt.xlim(xmax = stimulationtime * 1000 * 1.1) # xmax = 500)

plt.xlabel('time [ms]')
plt.ylabel('EMG potential [mV]')

plt.title('stimrep: %d, nervenum: %d, %s: %s' % (stimrep, nervenum, signalnames[nervenum], nervenames[nervenum]))

# <codecell>

nervedelays = {'left SLN': 0, # delay of the stimulation pulse trains in milliseconds
               'right SLN': 1,
               'left RLN': 2,
               'right RLN': 3,
               'left TA': 4,
               'right TA': 5}

# <codecell>

# right RLN is delayed by 3 ms, i.e. 3 * 25 samples = 75
# acquisition due to averaging happens only at end of one 25 KHz interval

# RLNdelay = 3e-3 * d.fs_EMG
# RLNdelay = 0

# <codecell>

try:
    nervedelay = nervedelays[stimulated_nerve]
except:
    print 'nervedelay not found in nervedelays dict'
    nervedelay = 0
    print 'use nervedelay = 0 !!!'
    
print 'nervedelay [millisec] = ', nervedelay

nervedelay_samples = nervedelay * d.fs_EMG / 1000.0
print 'nervedelay_samples = ', nervedelay_samples

# <codecell>

cleaned_EMG = d.allEMG[:, :, nervedelay_samples : (nervedelay_samples + Npulses * Nphase)]

# <codecell>

for nervenum in range(NEMG):
    plt.figure(nervenum)
    plt.plot(d.time_EMG[:Nend] * 1000, 
             d.allEMG[nervenum, stimrep, RLNdelay:Nend + RLNdelay].T)

    plt.title('stimulated: ' + os.path.split(datadir)[-1] + 
              ', recorded nerve: ' + nervenames[nervenum] + 
              ', stimulation#: ' + str(stimrep))

    plt.xlabel('time [ms]')
    plt.ylabel('EMG potential [V]')
    
plt.show()

# <codecell>

d.allps.shape

# <codecell>

plt.imshow(d.allps, aspect = 'auto')
cb = plt.colorbar()
cb.set_label('subglottal pressure [Pa]')
plt.xlabel('time [samples]')
plt.ylabel('stimulation index')

NpsQ = d.time_psQ.shape[0]
time_index = np.arange(0, NpsQ, np.int(200e-3 * d.fs_psQ))

plt.xticks(time_index, [np.int(item) for item in np.round(d.time_psQ[time_index] * 1000)])
plt.xlabel('time [ms]')

plt.yticks(np.arange(0, d.Nrecnums, 4))

# <codecell>

plt.plot(d.allps[9, :])

# <codecell>

plt.imshow(np.log10(d.allEMG[nervenum, :, :]**2), aspect = 'auto')
plt.title(nervenames[nervenum])

# <codecell>

nervenum = 3
stimrep = 20

print 'nervenum: ', nervenum
print 'stimrep: ', stimrep

s = d.allEMG[nervenum, stimrep, nervedelay_samples:nervedelay_samples + Nend]

maxpeakf = d.frep * 1.5

Nnopeaks = np.int(d.fs_EMG / maxpeakf)
print 'Nnopeaks: ', Nnopeaks

pp = d.find_peaks(s, neighbors = Nnopeaks)
pm = d.find_peaks(-s, neighbors = Nnopeaks)

# <codecell>

plt.plot(d.time_psQ, s)
plt.plot(d.time_psQ[pp], s[pp], 'ro')
plt.plot(d.time_psQ[pm], s[pm], 'go')

plt.title(nervenames[nervenum])
#plt.xlim(xmax = 5 * 2000)

# <codecell>

plt.plot(s[pp[:-1]], s[pp[1:]], 'r.')
plt.axis('equal')

# <codecell>

plt.plot(d.time_EMG[pp] * 1000, 
         s[pp] - s[pm])
plt.title('peak-to-peak')

# <codecell>

stimulationtime = 1.5 # in sec

Npulses = frep * stimulationtime
Nphase = np.round(d.fs_EMG / frep)
Nsamples_segment = Nphase

print Npulses
print Nphase

# <codecell>

cleaned_EMG = d.allEMG[:, :, RLNdelay : (RLNdelay + Npulses * Nphase)]

phase = cleaned_EMG.copy()

# reshape into segments for phase averaging
phase.shape = (NEMG, Nstimulations, Npulses, -1)

print d.allEMG.shape
print cleaned_EMG.shape
print phase.shape

# <codecell>

def log_square(signal, reverse = False, identity = False):
    if identity:
        return signal
    
    if not reverse:
        return 10. * np.log10( signal**2 + 1e-7)
    else:
        return np.sqrt( np.power(10, signal / 10.0) )

# <codecell>

nervenum = 0
stimrep = 20

phase_signal = log_square( phase[nervenum, stimrep, :, :] , identity = True)

# <codecell>

mean_sign = np.mean( np.sign( phase[nervenum, stimrep, :, :] ), axis = 0)

# <codecell>

plt.plot(mean_sign)
plt.plot(np.sign(mean_sign))

# <codecell>

outdict = plt.boxplot( phase_signal, # [nervenum, stimrep, :, :], 
                      sym = 'g.', whis = 1.5) # whisker: 1.5 * (75 - 25) percentile

plt.xticks(np.arange(start = 0, stop = Nsamples_segment, step = Nsamples_segment / 10))

# plt.ylim(ymin = -1, ymax = 1)

plt.ylabel('amplitude [dB]')
plt.title(nervenames[nervenum] + ', stimrep: ' + str(stimrep))

# <codecell>

def stat_filtering(phase_signal, whisker = 1.5, meanfilterrange = 'q1-q3'):
    
    medians = np.median( phase_signal, axis = 0)

    q1, q3 = np.percentile( phase_signal, q = [25, 75], axis = 0)

    whis = whisker

    # interquartile range
    iq = q3 - q1

    high_val = q3 + whis * iq
    low_val = q1 - whis * iq
    
    high_val.shape = (-1, Nsamples_segment)
    low_val.shape = (-1, Nsamples_segment)
    
    whiskhigh = np.nanmax( np.where(phase_signal <= high_val, phase_signal, np.nan), axis = 0)
    whisklow = np.nanmin( np.where(phase_signal >= low_val, phase_signal, np.nan), axis = 0)

    whisklow.shape = (-1, Nsamples_segment)
    whiskhigh.shape = (-1, Nsamples_segment)
    
    flierlow = np.where(phase_signal < whisklow, phase_signal, np.nan)
    flierhigh = np.where(phase_signal > whiskhigh, phase_signal, np.nan)

    if meanfilterrange not in ['q1-q3', 'whiskers']:
        meanfilterrange = 'q1-q3'
        
    if meanfilterrange == 'q1-q3':
        filt_low = q1
        filt_high = q3
    if meanfilterrange == 'whiskers':
        filt_low = low_val
        filt_high = high_val
    
    meanfiltered = np.nanmean(
                              np.where( 
                                       np.where(phase_signal >= filt_low, 
                                                phase_signal, 
                                                np.nan) <= filt_high, 
                                       phase_signal, 
                                       np.nan), 
                              axis = 0)
    
    return dict(medians = medians, q1 = q1, q3 = q3, whisklow = whisklow.T, whiskhigh = whiskhigh.T,
                flierlow = flierlow.T, flierhigh = flierhigh.T, meanfiltered = meanfiltered)

# <codecell>

from PCA import PCA

# <codecell>

phase[nervenum, stimrep, :, :].shape

# <codecell>

PCA(phase[nervenum, stimrep, :, :], centering = True)

# <codecell>

nervenum = 2

gains = [1e4, 1e4, 1e4, 100]
gains = [1e3, 1e3, 1e3, 100]

print 'EMG signal from nerve: ', nervenames[nervenum]
print 'gain: ', gains[nervenum]
print

plt.close('all')

for stimrep in range(Nstimulations):
    stats = stat_filtering(phase[nervenum, stimrep, :, :], meanfilterrange = 'whiskers')
    # stats.keys()

    plt.figure(str(stimrep))
               
    plt.plot(stats['whisklow'], 'k')
    plt.plot(stats['q1'], 'b')
    plt.plot(stats['medians'], 'r')
    plt.plot(stats['q3'], 'b')
    plt.plot(stats['whiskhigh'], 'k')
    
    plt.plot(stats['flierlow'], 'g.')
    plt.plot(stats['flierhigh'], 'g.')
    
    plt.plot(stats['meanfiltered'], 'yo', mec = 'yellow', mfc = 'yellow')
    
    plt.xlabel('') # time [samples]')
    plt.ylabel('amplitude') # [dB]')

    plt.xticks(np.arange(0, Nsamples_segment, Nsamples_segment / 10), [''])
    plt.xlim(xmax = Nsamples_segment)
    
    # plt.ylim(ymin = -0.25, ymax = 0.25)
    # plt.text(x = 1, y = 0.23, s = 'stimrep: ' + str(stimrep))
    
    # plt.title(nervenames[nervenum])
    
xticks = np.arange(0, Nsamples_segment, Nsamples_segment / 10)
xtickslabels = ['%.0f' % (item / d.fs_EMG * 1000.0) for item in xticks]

plt.xticks(xticks, xtickslabels)
plt.xlabel('time [ms]')

# <codecell>

nervenum = 3
stimrep = 20

phasemin = np.min(phase[nervenum, stimrep, :, :])
phasemax = np.max(phase[nervenum, stimrep, :, :])

Nbins = 50
allhist = np.ones((Nphase, Nbins)) * np.nan

for phasenum, phasepoints in enumerate( phase[nervenum, stimrep, :, :].T ):
    allhist[phasenum, :], bins = np.histogram(phasepoints, bins = Nbins, normed = True, range = (phasemin, phasemax)) 

# <codecell>

plt.imshow(np.where(allhist > 0, allhist, np.nan).T, aspect = 'auto')

# plt.yticks([str(item) for item in bins[:-1:10]])

plt.ylabel('EMG potential [V]')
plt.colorbar()

# <codecell>

print stimrep
print nervenum

# <codecell>

stimrep = 20

for nervenum, nervename in enumerate(nervenames):
    plt.figure(nervename)
    
    plt.plot(d.time_EMG[:Nphase] * 1000,
             log_square( phase[nervenum, stimrep, :, :] ).T, '-')
    
    plt.plot(d.time_EMG[:Nphase] * 1000, 
             np.nanmean( log_square( phase[nervenum, stimrep, :, :] ), axis = 0), 
             color = 'red', marker = 'o', ms = 10, mec = 'yellow', mfc = 'red', ls = '-', lw = 5)
    
    plt.text(x = 25, y = 15, 
             s ='stimulated: ' + os.path.split(datadir)[-1] + '\nrecorded EMG: ' + nervename + '\nstimrep#: ' + str(stimrep),
             fontsize = 20,
             verticalalignment = 'top')
    
    plt.ylabel('amplitude [dB]')
    plt.ylim(ymin = -75, ymax = 20)
    
    plt.xlabel('')
    # plt.xticks(np.arange(10), [''] * 10)
    
# plt.xticks(np.arange(10), np.arange(10))
plt.xlabel('time [ms]')
    
plt.show()

# <codecell>

from mpl_toolkits.mplot3d import Axes3D

# <codecell>

ax = plt.subplot(111, projection = '3d')

ax.plot(d.time_EMG[:Nphase] * 1000, np.arange(Npulses), phase[nervenum, stimrep, :, :].T)

# <codecell>

plt.plot(# d.time_EMG[:Nphase] * 1000,
         np.std(phase[nervenum, stimrep, :, :], axis = 0))
plt.xlim(xmax = Nphase)
plt.show()

# <codecell>

nervenum = 3
stimrep = 20

plt.imshow(log_square(phase[nervenum, stimrep, :, :]), origin = 'upper', aspect = 'auto')

plt.xlabel('time [samples]')
plt.ylabel('pulse#')

plt.title('nervenum: ' + str(nervenum) + ': ' + nervenames[nervenum] + ', stimrep: ' + str(stimrep))

plt.colorbar()
# plt.show()

# <codecell>

import matplotlib.mlab as mlab

Pac, freq, time = mlab.specgram(cleaned_EMG[3, 10, :], NFFT = Nsamples_segment, Fs = d.fs_EMG, noverlap = 0, 
                                pad_to = 16 * Nsamples_segment)

# <codecell>

Pac_dB = 10 * np.log10(Pac)

# <codecell>

plt.pcolormesh(time * 1000, freq, Pac_dB)
plt.xlabel('time [ms]')
plt.ylabel('frequency [Hz]')

plt.colorbar()

plt.xlim(xmax = 1500)
plt.ylim(ymax = 1500)

# <markdowncell>

# Old code
# ----------

# <codecell>

%pylab inline
# %pylab
%autocall 2

# <codecell>

%cd /extra/public/python/tools/
import dogdata

# <codecell>

datadir = "/mnt/bucket/InVivoDog_2012_03_14/data LabView/EMG"

# <codecell>

import os, glob
filelist = glob.glob(os.path.join(datadir, '*.hdf5'))
print sorted(filelist)

# <codecell>

d = dict()

for filenum, filename in enumerate(filelist):
    print "case: ", filename
    datadirname = os.path.dirname(filename)
    datafilename = os.path.basename(filename)
    d[datafilename] = dogdata.DogData(datadir = datadirname, datafile = datafilename)
    print ""

# <codecell>

for dat in d.values():
    dat.get_all_data()

# <codecell>

for stimcase in sorted(d.keys()):
    figure()
    plot(d[stimcase].time_psQ, d[stimcase].allps.T)
    xlabel('time [sec]')
    ylabel('subglottal pressure [Pa]')
    title(stimcase)
    legend(["%d" % (k + 1) for k in range(20)], loc = 'upper left', bbox_to_anchor = (1.0, 1.0))

# <codecell>

for stimcase in sorted(d.keys()):
    plot(np.max(d[stimcase].allps, axis = 1), lw = 2, label = stimcase)

# ylim(ymin = -10)
xlabel('stimulation index')
ylabel('maximum subglottal pressure [Pa]')
legend(loc = 'upper left', bbox_to_anchor = (1.0, 1.0))
# title(d.datafilename)

# <codecell>

EMGchannel = 3
for stimcase in sorted(d.keys()):
    plot(np.std(d[stimcase].allEMG, axis = 2)[EMGchannel - 1, :] * 1000, lw = 2, label = stimcase)
    
ylabel('EMG std [mV]')
legend(loc = 'upper left', bbox_to_anchor = (1.0, 1.0))
title('EMG channel: %s' % EMGchannel)

# <codecell>

for stimcase in sorted(d.keys()):
    figure()
    plot(np.std(d[stimcase].allEMG, axis = 2).T * 1000, lw = 2, label = stimcase)
    ylabel('EMG std [mV]')
    title("%s" % stimcase)
    legend(["%d" % (k + 1) for k in range(20)], loc = 'upper left', bbox_to_anchor = (1.0, 1.0))
    ylim(ymin = 0, ymax = 700)

# <codecell>

ts = 500 * 2 # time in samples
for stimcase in sorted(d.keys()):
    figure()
    plot(d[stimcase].time_EMG[:ts] * 1000, d[stimcase].allEMG[2, 6, :ts] * 1000, 'r', lw = 2, label = stimcase)
    xlabel('time [millisec]')
    ylabel('EMG [mV]')
    title("%s" % stimcase)
    legend(["EMG 3"], loc = 'upper left', bbox_to_anchor = (1.0, 1.0))
    # ylim(ymin = 0, ymax = 700)

# <codecell>


