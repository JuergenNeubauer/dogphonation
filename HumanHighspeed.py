# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, sys, glob
import numpy as np
sys.path.append('/extra/InVivoDog/python/cine/tools')

import readTDMS as tdms
import nptdms

# <codecell>

import scipy.io.wavfile as wavfile

# <codecell>

for dirpath, dirnames, filenames in os.walk('/scratch/HumanHighspeed/'):
    for filename in filenames:
        fullfilepath = os.path.join(dirpath, filename)
        
        if not fullfilepath.endswith('.converted.tdms'):
            continue

        if os.path.isfile(fullfilepath.replace('.converted.tdms', '.wav')):
            continue
            
        casedir = dirpath
        casename = filename
        
        try:
            metadata, rawchanneldata, datafilename = tdms.readfile(casedir, casename)
        except Exception as e:
            print e
            print "problems reading: "
            print casedir
            print casename
            print "filesize of original tdms [Bytes]: ", os.path.getsize(fullfilepath.replace('.converted.tdms', '.tdms'))
            print
            continue
        print
        
        savefilepath = fullfilepath.replace('.converted.tdms', '.metadata.npz')
        
        print "saving metadata: ", savefilepath
        np.savez_compressed(savefilepath, metadata = metadata)
        
        channelnames = rawchanneldata.keys()
        
        if len(channelnames) > 1:
            print "more than ONE channel found"
            print "IGNORING"
            print
            continue
            
        rawdata = np.array(rawchanneldata[channelnames[-1]])
        
        try:
            samplerate = np.round(1 / metadata["/'outside audio'/'cDAQ1Mod1/ai0'"][3]['wf_increment'][1])
        except Exception as e:
            print e
            print "problem reading metadata"
            print metadata
            print
            
        try:
            samplerate = np.round(1 / metadata["/'outside audio'/'microphone'"][3]['wf_increment'][1])
        except Exception as e:
            print e
            print "problem reading metadata"
            print metadata
            print
            continue
            
        smin = np.abs(rawdata.min())
        smax = np.abs(rawdata.max())
        sabsmax = np.max( [smin, smax] )
        
        s_scale = (rawdata.astype(np.float64) / sabsmax * 0.99 * (2**15 - 1)).astype(np.int16)
        
        wavfilename = fullfilepath.replace('.converted.tdms', '.wav')
        
        print "saving wav file: ", wavfilename
        print "sampling rate [Hz]: ", samplerate
        wavfile.write(wavfilename, samplerate, s_scale)
        
        print

# <codecell>

for key in sorted(metadata.keys()):
    print 'key: ', key
    print 'data: ', metadata[key]
    print

# <codecell>

np.round(1 / metadata["/'outside audio'/'cDAQ1Mod1/ai0'"][3]['wf_increment'][1])

# <codecell>

channelnames = rawchanneldata.keys()

# <codecell>

channelnames

# <codecell>

rawdata = np.array(rawchanneldata[channelnames[0]])

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib.pyplot as plt

# <codecell>

plt.plot(rawdata)

# <codecell>

import aubio

# <codecell>

import pyaudio

# <codecell>

help pyaudio.Stream

# <codecell>

help pyaudio.PyAudio

# <codecell>

help pyaudio.PyAudio.open

# <codecell>

# needs to be activated at startup
%gui pyglet

# <codecell>

import pyglet

# <codecell>

datadir = "/scratch/HumanHighspeed/HGG_2013_04_05/S31"
wavfile = r"S31 breathy_high Fri Apr 05 2013 15 30 44.wav"

fullwavfile = os.path.join(datadir, wavfile)

# <codecell>

sound = pyglet.media.load(fullwavfile, streaming = False) # streaming = False: preload data for faster playing
# sound = pyglet.resource.media('shot.wav', streaming = False)
sound.play()

# <codecell>

# from github: ipython / examples / lib / gui-pyglet.py 
try:
    from IPython.lib.inputhook import enable_pyglet
    enable_pyglet()
except ImportError:
    pyglet.app.run()

# <codecell>

import IPython.kernel.zmq as zmq

# <codecell>

zmq.datapub.publish_data?

# <codecell>


