# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob, os, sys, subprocess, shlex
import numpy as np

import paramiko

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')
import cine

# <codecell>

%matplotlib inline

import matplotlib.pyplot as plt

# <codecell>

ssh_client = paramiko.SSHClient()

ssh_client.load_system_host_keys()

ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# <codecell>

with open('/home/neubauer/.neurousername', 'r') as f:
    username = f.read()

if not username:
    print "no username"
    username = None
else:
    print 'username: ', username
    
with open('/home/neubauer/.neuropassword', 'r') as f:
    password = f.read()
    
if not password:
    print "no password"
    print
    
    try:
        password = raw_input("Enter password for users '%s': " % username)
    except:
        print "some problem occurred"
        print sys.exc_info()
        password = None
    else:
        print "got a password"
else:
    print "got a password"

# <codecell>

neuromuscular = '10.47.85.12' # must be inside the VPN network

ssh_client.connect(neuromuscular, username = username, password = password, compress = True)

# <codecell>

remotecinefilepath = ("'/mnt/workspace/InVivoDog_2013_12_11/asymmetric RLN/No SLN/'" +
                      "'asymmetric RLN No SLN_092_Wed Dec 11 2013 13 14 17.007 809.001.cine'")
ssh_in, ssh_out, ssh_err = ssh_client.exec_command("ipython /extra/public/python/cinekymo.py " + remotecinefilepath + ' 250')

r_out = ssh_out.readlines()
r_err = ssh_err.readlines()

# <codecell>

for line in r_out: print line.strip()

# <codecell>

for line in r_err: print line.strip()

# <codecell>

ftp = ssh_client.open_sftp()

# <codecell>

ftp.chdir(path = '.')
ftp.getcwd()

# <codecell>

remotepath = "/mnt/workspace/InVivoDog_2013_12_11/asymmetric RLN/No SLN"

ftp.chdir(path = remotepath)
print ftp.getcwd()
print 
cinenames = [item for item in ftp.listdir() if item.endswith('.cine')]

# <codecell>

remotefilename = cinenames[0]

ftpcinefile = ftp.open(filename = remotefilename, 
                       mode = 'r', 
                       bufsize = 512 * 512 * 2) # might need to use large bufsize to hold one frame ???

# <codecell>

from cinefile_datatypes import *

# <codecell>

cinefileheader_type.itemsize

# <codecell>

cinefileheader = np.frombuffer(ftpcinefile.read(size = cinefileheader_type.itemsize), 
                               count = 1, 
                               dtype = cinefileheader_type)

# <codecell>

dict( zip(cinefileheader_type.names, cinefileheader.item() ) )

# <codecell>

ftpcinefile.seek(0)

# <codecell>

help ftpcinefile.readv

# <codecell>


