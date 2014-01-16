# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

# <codecell>

from IPython.display import Audio
Audio(url="http://www.nch.com.au/acm/8k16bitpcm.wav")

# <codecell>

import scipy.constants as const
import scipy
from scipy.io import wavfile
from IPython.core.display import display, HTML
from __future__ import division

# <codecell>

# this is a wrapper that take a filename and publish an html <audio> tag to listen to it

import sys
import StringIO
import base64

def wavPlayer(data, rate):
    """ will display html 5 player for compatible browser

    The browser need to know how to play wav through html5.

    there is no autoplay to prevent file playing when the browser opens
    
    Adapted from SciPy.io.
    """
    
    buffer = StringIO.StringIO()
    buffer.write(b'RIFF')
    buffer.write(b'\x00\x00\x00\x00')
    buffer.write(b'WAVE')

    buffer.write(b'fmt ')
    if data.ndim == 1:
        noc = 1
    else:
        noc = data.shape[1]
    bits = data.dtype.itemsize * 8
    sbytes = rate*(bits // 8)*noc
    ba = noc * (bits // 8)
    buffer.write(struct.pack('<ihHIIHH', 16, 1, noc, rate, sbytes, ba, bits))

    # data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<i', data.nbytes))

    if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
        data = data.byteswap()

    buffer.write(data.tostring())
#    return buffer.getvalue()
    # Determine file size and place it in correct
    #  position at start of the file.
    size = buffer.tell()
    buffer.seek(4)
    buffer.write(struct.pack('<i', size-8))
    
    val = buffer.getvalue()
    
    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>
    
    <body>
    <audio controls="controls" style="width:600px" >
      <source controls src="data:audio/wav;base64,{base64}" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """.format(base64=base64.encodestring(val))
    display(HTML(src))

# <codecell>

## some consstant for our audio file 

rate = 44100 #44.1 khz
duration =2.5 # in sec

# this will give us sin with the righ amplitude to use with wav files
normedsin = lambda f,t : 2**13*np.sin(2*np.pi*f*t)

time = np.linspace(0,duration, num=rate*duration)

# <codecell>

from __future__ import division, print_function, absolute_import

import numpy
import struct
import warnings

# <codecell>

# define A as a 440 Hz sin function 
la    = lambda t : normedsin(440,t)

# look at it on the first 25 ms
# plot(time[0:1000], la(time)[0:1000])

ampl = la(time).astype(np.int16)

# write the file on disk, and show in in a Html 5 audio player
wavPlayer(ampl, rate)

# <codecell>


