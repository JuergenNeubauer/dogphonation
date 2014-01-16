# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob, os, subprocess, shlex
import numpy as np

# <codecell>

%matplotlib inline

import matplotlib as mpl
# mpl.use('module://IPython.zmq.pylab.backend_inline')
import matplotlib.pyplot as plt

# <codecell>

pwd

# <codecell>

datadir = "../python/cine/results_save/"

# <codecell>

TAconditions = ["NoTA", "ThresholdTA_condition01", 'TA_condition02', 'TA_condition03', 'MaxTA_condition04']

# <codecell>

landmarkfiles = {}
TAnumbers = []

for TAindex, TA in enumerate(TAconditions):
    filenames = sorted(glob.glob(os.path.join(datadir, 'SLN_trunkRLN_' + TA + '*.png')))
    
    TAnumbers.append('TA %d' % TAindex)
    
    landmarkfiles[TAnumbers[TAindex]] = dict(TAcondition = TA, files = filenames)

# <codecell>

# the first condition for a given TA only used TA and no other nerve
onlyTAstimulation_images = [landmarkfiles[TAindex]['files'][0] for TAindex in TAnumbers]

# <codecell>

r = subprocess.check_output(shlex.split("""identify -format '%%w\n%%h' "%s" """ % onlyTAstimulation_images[0]))

im_width, imheight = [int(item) for item in r.splitlines()]

# <codecell>

# geometry for inline crop
width = 600
height = 940
xoffset = 1500
yoffset = 100

# <codecell>

# geometry for inline crop
width = im_width / 4
height = 880 # im_height

xoffset = im_width * 5 / 8
yoffset = 170

croparea_inline = "[%dx%d+%d+%d]" % (width, height, xoffset, yoffset)

# <codecell>

image_names = [' "%s"%s ' % (imagename, croparea_inline) for imagename in onlyTAstimulation_images]

# <codecell>

annotate_opts = " ".join(["-gravity northwest",
                          "-fill white",
                          "-pointsize 100"])

# <codecell>

image_cmd = " ".join(["( %s +repage -annotate +10+10 'TA %d' ) " % (name, imnum) for imnum, name in enumerate(image_names)])

# <codecell>

brighten_cmd = " ".join(["-contrast",
                         # "-contrast",
                         "-equalize"])

border_cmd = " ".join(["-bordercolor white -border 5"])

convert_cmd = " ".join(["convert", # "-respect-parenthesis",
                        annotate_opts,
                        image_cmd,
                        brighten_cmd,
                        border_cmd,
                        "+append png:TAatOnset.png"])

# <codecell>

subprocess.check_call(shlex.split(convert_cmd), shell = False)

# <codecell>

cinenames = [os.path.basename(item).replace('.landmarks.png', '') for item in onlyTAstimulation_images]

# <codecell>

r = subprocess.check_output(shlex.split("locate '%s'" % cinenames[0])).splitlines()
r

# <codecell>

straindata = np.load(r[1])

# <codecell>

for item in straindata.files:
    print item, ": ", straindata[item]

# <codecell>

strainfiles = [item.replace('landmarks.png', 'strain_onset.npz') for item in onlyTAstimulation_images]

# <codecell>

[np.load(item)['onsetframenumber'].item() for item in strainfiles]

# <codecell>

import paramiko

# <codecell>

ssh_client = paramiko.SSHClient()

ssh_client.load_system_host_keys()

ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# <codecell>

neuromuscular = '10.47.85.12'
ssh_client.connect(neuromuscular, username = 'neubauer', password = "Qy72:<>z", compress = True)

# <codecell>

ftp = ssh_client.open_sftp()

# <codecell>

ftp.chdir(path = '.')
ftp.getcwd()

# <codecell>

# ftp.open(filename = , mode = 'r', bufsize = -1)

# <codecell>

cinenames

# <codecell>

s = cinenames[-1]

# <codecell>

ssh_in, ssh_out, ssh_err = ssh_client.exec_command("locate '*SLN_trunkRLN_MaxTA_condition04*.cine'")

r = sorted([line for line in ssh_out.read().splitlines()])

# <codecell>

[item for item in r if os.path.basename(item).replace('.cine', '').find("Wed Mar 21 2012 15 19") > 0]

# <codecell>

import re

# <codecell>

help re

# <codecell>

print ssh_err.readlines()

# <codecell>

line

# <codecell>

ftp.get(remotepath = line, localpath = os.path.basename(line))

# <codecell>

ftp.close()

# <codecell>

ssh_client.close()

# <codecell>


