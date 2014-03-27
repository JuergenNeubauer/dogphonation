# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import subprocess, shlex

# <markdowncell>

# sox -V original.wav test_sox.wav tempo  -s 0.25
# has Python interface
# 
# marsyas: 
# phasevocoder original.wav -f test_marsyas.wav -n 2048 -w 2048 -d 128 -i 512 -cm full
# 
# rubberband:
# rubberband -d3 --time 4 --window-short original.wav test.wav
# 
# soundstretch:
# soundstretch trunkRLN_TA_NoSLN\ Wed\ Mar\ 21\ 2012\ 17\ 18\ 17.016.psub.wav test.wav -tempo=-70
# 
# csound: pitch analysis
# lpanal -a -P30 -Q500 -v 2 original.wav test_lpanal
# csound has Python interface

# <markdowncell>

# https://github.com/paulnasca/paulstretch_python/blob/master/paulstretch_stereo.py

# <codecell>

!soundstretch

# <codecell>

!soundstretch infile outfile -speech -tempo=-75

# <codecell>

!man rubberband

# <codecell>

!rubberband

# <codecell>

!phasevocoder --help

# <markdowncell>

# http://marsyas.info/assets/docs/manual/marsyas-user/phasevocoder.html#phasevocoder

# <codecell>

!phasevocoder original.wav -f test_marsyas.wav --fftsize 2048 --winsize 2048 --decimation 128 --interpolation 512 -cm full

# <codecell>

ls /opt/aubio/aubio-0.4.0/python/demos/

# <codecell>

!man soxeffect

# <codecell>

import pysox

# <codecell>

soxapp = pysox.CSoxApp('original.wav', 'test_sox.wav')

# <codecell>

soxapp.flow([ ('tempo', [b'-s', b'0.25']) ])

