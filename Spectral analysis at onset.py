# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline

# <codecell>

import numpy as np
import os, sys, glob

import matplotlib as mpl
import matplotlib.pyplot as plt

# <codecell>

import scipy.io.wavfile as wavfile

# <codecell>

datadir = '/extra/InVivoDog/InVivoDog_2012_03_21/data LabView/SLN_trunkRLN/'
wavfiledir = 'SLN_trunkRLN_ThresholdTA_condition01 Wed Mar 21 2012 14 55 08.wavfiles'

# <codecell>

dirpath = os.path.join(datadir, wavfiledir)

# <codecell>

allwavfiles = sorted(glob.glob(dirpath + '/*.psub.wav'))

# <codecell>

chest_wavfilename = allwavfiles[2]
falsetto_wavfilename = allwavfiles[-3]

print "chest: ", chest_wavfilename
print 
print "falsetto: ", falsetto_wavfilename

# <codecell>

!play "$falsetto_wavfilename"

# <codecell>

backgroundnoisefile = os.path.join(datadir, 
                                   'SLN_trunkRLN_NoTA Wed Mar 21 2012 14 46 34.wavfiles',
                                   'SLN_trunkRLN_NoTA Wed Mar 21 2012 14 46 34.001.psub.wav')

# <codecell>

fs, background = wavfile.read(backgroundnoisefile)

# <codecell>

fs, chest = wavfile.read(chest_wavfilename)
fs, falsetto = wavfile.read(falsetto_wavfilename)

# <codecell>

plt.plot(chest, '-')

plt.xlim(xmin = 3000, xmax = 20000)

# <codecell>

Nfft = 4096
overlap = 4000
hop = Nfft - overlap

# <codecell>

chestPxx, freqs, timebins, specgramimage = plt.specgram(chest, NFFT = Nfft, Fs = fs, noverlap = overlap, 
                                                        origin = 'upper')

plt.ylim(ymax = 1000)

# <codecell>

plt.imshow(10 * np.log10(chestPxx), aspect = 'auto')

plt.xlim(xmax = 200)

plt.ylim(ymax = np.nanargmax(np.where(freqs < 1000, freqs, np.nan)))

# <codecell>

slice_chest = 120

plt.plot(freqs, 10 * np.log10(chestPxx[:, slice_chest]))

plt.xlim(xmax = 2000)

# <codecell>

plt.plot(falsetto, '-')

plt.xlim(xmin = 45000, xmax = 52000)

# <codecell>

falsettoPxx, freqs, timebins, specgramimage = plt.specgram(falsetto, NFFT = Nfft, Fs = fs, noverlap = overlap, 
                                                           origin = 'upper')

plt.ylim(ymax = 2000)

# <codecell>

plt.imshow(10 * np.log10(falsettoPxx), aspect = 'auto')

plt.xlim(xmin = 450, xmax = 650)

plt.ylim(ymax = np.nanargmax(np.where(freqs < 2000, freqs, np.nan)))

# <codecell>

slice_falsetto = 570

# plt.plot(freqs, 10 * np.log10(backgroundPxx[:, slice_background]) + 10, '-', label = 'background')

plt.plot(np.log2(freqs), 10 * np.log10(chestPxx[:, slice_chest] / np.max(chestPxx[:, slice_chest])), 
         '-', label = 'chest')

plt.plot(np.log2(freqs), 10 * np.log10(falsettoPxx[:, slice_falsetto] / np.max(falsettoPxx[:, slice_falsetto])), 
         '-', label = 'falsetto')

plt.xlim(xmin = 6,
         xmax = np.log2(5000))

plt.ylim(ymin = -100)

plt.legend(loc = 'lower left')

plt.xlabel('frequency [log2(Hz)]')
plt.ylabel('power [dB]')

plt.savefig('chest_falsetto_spectra.pdf', orientation = 'landscape', bbox_inches = 'tight')

# <codecell>

backgroundPxx, freqs, timebins, specgramimage = plt.specgram(background, NFFT = Nfft, Fs = fs, noverlap = overlap, 
                                                        origin = 'upper')

plt.ylim(ymax = 2000)

# <codecell>

plt.imshow(10 * np.log10(backgroundPxx), aspect = 'auto')

# plt.xlim(xmin = 450, xmax = 550)

plt.ylim(ymax = np.nanargmax(np.where(freqs < 2000, freqs, np.nan)))

# <codecell>

slice_background = 500

plt.plot(freqs, 10 * np.log10(np.mean( backgroundPxx[:, 200:700] , axis = 1)), '-', label = 'background')

plt.xlim(xmax = 5000)
plt.ylim(ymin = -20)

plt.xlabel('frequency [Hz]')
plt.ylabel('power [dB]')

plt.savefig('backgroundnoise_subglottalresonance.pdf', orientation = 'landscape', bbox_inches = 'tight')

# <codecell>

import audiolazy as lz

# <codecell>

s, Hz = lz.sHz(fs)
print s, Hz

# <codecell>

print "Nfft: {}, overlap: {}, hop: {}".format(Nfft, overlap, hop)

stream_chest = lz.Stream(chest).blocks(size = Nfft, hop = hop)
stream_falsetto = lz.Stream(falsetto).blocks(size = Nfft, hop = hop)

stream_background = lz.Stream(background).blocks(size = Nfft, hop = hop)

for k in range(slice_chest):
    # skip the first Nskip blocks of data
    stream_chest.take()
    
for k in range(slice_falsetto):
    stream_falsetto.take()
    
for k in range(slice_background):
    stream_background.take()
    
data_chest = stream_chest.take()
data_falsetto = stream_falsetto.take()
data_background = stream_background.take()

data_chest = lz.resample(data_chest, old = 4, new = 1).take(lz.inf)

data_falsetto = lz.resample(data_falsetto, old = 2, new = 1).take(lz.inf)

data_background = lz.resample(data_background, old = 4, new = 1).take(lz.inf)

# <codecell>

plt.plot(data_chest, label = 'chest')
plt.plot(data_falsetto, label = 'falsetto')

plt.plot(data_background, label = 'background')
plt.legend()

print len(data_chest)

print "new sampling frequency: ", fs / 4.0

s, Hz = lz.sHz(fs / 4.0)

# <codecell>

lpc_chest = lz.lpc(data_chest, order = 14)
lpc_falsetto = lz.lpc(data_falsetto, order = 7)

lpc_background = lz.lpc(data_background, order = 14)

print lpc_chest.error
print lpc_falsetto.error
print lpc_background.error

# <codecell>

print lpc_chest

# <codecell>

Hz == (2 * lz.pi / fs)

# <codecell>

plt.close('all')

mpl.interactive(False)
plt.interactive(False)
plt.ioff()

maxf = 5000.0

# fig = (1 / lpc_chest).plot(rate = s * 1, samples = maxf / fs * Nfft, unwrap = False, max_freq = maxf * Hz / 1)

# fig = (1 / lpc_falsetto).plot(fig = fig, rate = s * 2, samples = maxf / fs * Nfft, unwrap = False, max_freq = maxf * Hz / 2)

fig = (1 / lpc_background).plot(# fig = fig, 
                                rate = s * 1, samples = maxf / fs * Nfft, unwrap = False, max_freq = maxf * Hz / 1)

fig.axes[0].plot(freqs, 10 * np.log10(np.mean( backgroundPxx[:, 200:700] , axis = 1)), '-', label = 'background')

for a in fig.axes:
    for l in a.lines:
        l.set_marker('None')
        
plt.savefig('lpc_background.pdf', orientation = 'landscape', bbox_inches = 'tight')

# <codecell>

plt.close('all')
plt.ioff()

# calculates the DFT the VERY slow way with discrete complex function projections
fig = (1e1 / lpc_background).plot(blk = data_chest, 
                           rate = s * 1, samples = maxf / fs * Nfft, unwrap = False, max_freq = maxf * Hz / 1)

for a in fig.axes:
    for l in a.lines:
        l.set_marker('None')

# <codecell>

plt.plot(freqs, 10 * np.log10(chestPxx[:, slice_chest]), '-', label = 'chest')

# plt.plot(freqs, 10 * np.log10(falsettoPxx[:, slice_falsetto]), '-', label = 'falsetto')

plt.xlim(xmax = 5000)

plt.ylim(ymin = -40)

plt.legend()

plt.xlabel('frequency [Hz]')
plt.ylabel('power [dB]')

# <codecell>

IF_chest = lpc_background(data_chest).take(lz.inf)
IF_falsetto = lpc_background(data_falsetto).take(lz.inf)

# <codecell>

PxxIF_chest, f_IF_chest, t_IF = mpl.mlab.specgram(IF_chest, NFFT = Nfft / 4, Fs = fs / 4, noverlap = 0)
PxxIF_falsetto, f_IF_falsetto, t_IF = mpl.mlab.specgram(IF_falsetto, NFFT = Nfft / 2, Fs = fs / 2, noverlap = 0)

# <codecell>

# plt.plot(freqs, 10 * np.log10(chestPxx[:, slice_chest]), '-', label = 'chest')
# plt.plot(freqs, 10 * np.log10(falsettoPxx[:, slice_falsetto]), '-', label = 'falsetto')

plt.plot(np.log2(f_IF_chest), 10 * np.log10(PxxIF_chest / np.max(PxxIF_chest)), '-', label = 'filtered chest', )
plt.plot(np.log2(f_IF_falsetto), 10 * np.log10(PxxIF_falsetto / np.max(PxxIF_falsetto)), '-', label = 'filtered falsetto')

plt.xlim(xmin = 6,
         xmax = np.log2(5000))

plt.ylim(ymin = -100)

plt.legend(loc = 'lower left')

plt.xlabel('frequency [log2(Hz)]')
plt.ylabel('power [dB]')

plt.savefig('filtered_chest_falsetto_spectra.pdf', orientation = 'landscape', bbox_inches = 'tight')

# <codecell>

blabal bal

