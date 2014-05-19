# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, sys, glob
import numpy as np

# <codecell>

%matplotlib inline

# <codecell>


# <codecell>

%matplotlib inline

# <codecell>

import matplotlib as mpl
import matplotlib.pyplot as plt

# <codecell>

!play robot.wav

# <codecell>

import aubio

# <codecell>

dirname = "../SLN_trunkRLN_TA_condition02 Wed Mar 21 2012 15 01 30.wavfiles/"
filename = "SLN_trunkRLN_TA_condition02 Wed Mar 21 2012 15 01 30.032.psub.wav"

fullfilename = os.path.join(dirname, filename)

# <codecell>

win_s = 1024
hop_s = win_s / 4

samplerate = 0

s = aubio.source(fullfilename, samplerate, hop_s)

# <codecell>

print s.samplerate
print s.hop_size

# <codecell>


# <codecell>

!play "$fullfilename"

# <codecell>


# <codecell>

fs = 50000
samplerate = fs

# <codecell>

windowsize = 4096
buffersize = windowsize

hopsize = windowsize / 2

min_freq = 30
hopsize = int(fs / float(min_freq) * 2.0)

print hopsize

# <codecell>

methods = ['default', 'schmitt', 'fcomb', 'mcomb', 'yin', 'yinfft']

# <codecell>

(6 - 4 + 1) / (4 - 2)

# <codecell>

def run_pitchtracker(pitchtracker, signal, overlap = 0):
    
    hop_size = pitchtracker.hop_size
    
    if overlap < 0:
        overlap = 0
    
    if overlap > 0:
        overlap = min(overlap, hop_size - 1)
        
        advance = hop_size - overlap
        
        Nwindow = (len(signal) - hop_size + 1) / advance
    
    if overlap == 0:
        advance = hop_size
        
        Nwindow = len(signal) / hop_size # integer division
        
        samples = signal[:Nwindow * hop_size].reshape((-1, hop_size))
    
    sample_buffer = aubio.fvec(hop_size)
    
    pitch = np.empty(Nwindow)
    confidence = np.empty(Nwindow)
    
    if overlap == 0:
        # process in parallel on different cores
        for buffernum, sample in enumerate(samples):
            sample_buffer[:] = sample
            
            pitch[buffernum] = pitchtracker(sample_buffer)[0]
            confidence[buffernum] = pitchtracker.get_confidence()
    elif overlap > 0:
        for buffernum in range(Nwindow):
            start = buffernum * advance
            sample_buffer[:] = signal[start:start + hop_size]
            
            pitch[buffernum] = pitchtracker(sample_buffer)[0]
            confidence[buffernum] = pitchtracker.get_confidence()
        
    return pitch, confidence, advance

# <codecell>

pitchtrackers = {}

for method in methods:
    pitchtrackers[method] = aubio.pitch(method, buffersize, hopsize, samplerate)
    
    pitchtracker = pitchtrackers[method]
    
    # level threshold under which pitch should be ignored, in dB
    # pitchtracker.set_silence(-50) # default: -50 dB
    
    if method is "yin":
        tolerance = 0.15
    elif method is "yinfft":
        tolerance = 0.85
    else:
        tolerance = 0.6
        
    pitchtracker.set_tolerance(tolerance)
    pitchtracker.set_unit("freq")
    
    print pitchtracker.method
    print pitchtracker.get_silence()
    print pitchtracker.get_confidence()

# <codecell>

pitchtrackers

# <codecell>

import scipy.io.wavfile as wavfile

# <codecell>

rate, signal = wavfile.read(fullfilename)

# <codecell>

overlap = hopsize / 1

pitch, confidence, advance = run_pitchtracker(pitchtrackers['yinfft'], signal, overlap = overlap)

print pitch.shape

print advance

# <codecell>

plt.specgram(signal, NFFT = 4096, Fs = fs, noverlap = 4000, origin = 'upper')

plt.plot(np.arange(len(pitch)) * advance / 50000., np.where(confidence > 0.8, pitch, np.nan))
         
plt.ylim(ymax = 1000)

# plt.xlim(xmin = 0.2, xmax = 0.5)

# <codecell>

plt.plot(np.arange(len(pitch)) * advance / 50000., 
         pitch)
plt.ylim(ymax = 1000)

# <codecell>

plt.plot(np.arange(len(pitch)) * advance / 50000.0, confidence)

# <codecell>

source = aubio.source(filename, fs, hopsize)

# <codecell>

pv = aubio.pvoc(windowsize, hopsize)

# <codecell>

samples, read = source()

# <codecell>

print samples.shape
print read

# <codecell>

spectrum = pv(samples)

print spectrum.norm.shape
print spectrum.phas.shape

# <codecell>

plt.close('all')

# plt.plot(np.log10(spectrum.norm))
plt.plot(spectrum.phas * 180 / np.pi)

# <codecell>


