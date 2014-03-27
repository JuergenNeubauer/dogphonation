# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

globals

# <codecell>

from IPython.parallel import Client

# <codecell>

with open('/home/neubauer/.neurousername', 'rt') as f:
    # suppress newline \n character
    username = f.read()[:-1]
with open('/home/neubauer/.neuropassword', 'rt') as f:
    password = f.read()[:-1]

# <codecell>

try:
    del c
except:
    pass

# <codecell>

c = Client(profile = 'parallel', cluster_id = 'anaconda')

# <codecell>

c = Client(profile = 'parallel', cluster_id = 'ipython')

# <codecell>

c.ids

# <codecell>

e0 = c[0]

# <codecell>

e0.block = True
e0.activate(suffix = '0anaconda')

# <codecell>

%%px0anaconda
X = np.random.random((1000, 3))

# <codecell>

%%px0anaconda
def pairwise_numpy(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))
%timeit pairwise_numpy(X)

# <codecell>

%%px0anaconda
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D
%timeit pairwise_python(X)

# <codecell>

%%px0anaconda
from numba import double
from numba.decorators import jit, autojit

pairwise_numba = autojit(pairwise_python)

%timeit pairwise_numba(X)

# <codecell>

import cinefile_datatypes

# <codecell>

# size of image of 10 bit pixels in bytes
imagesize = 512 * 512 * 10 / 8
print imagesize

# <codecell>

framecount = 4501
height = 512
width = 512

# <codecell>

# pixdat = np.fromfile(self.file, count = 1, dtype = self.allframes_type)

frame_type = np.dtype([('annotationsize', np.uint32),
                       ('imagesize', np.uint32),
                       ('pixel', np.uint8, imagesize),
                       ])

Nframes_type = np.dtype([('Nframes', 
                          frame_type, 
                          framecount)])

# pixdat = np.fromfile(self.file, count = 1, dtype = Nframes_type)
pixdat = np.empty(1, dtype = Nframes_type)

# pixdat =  pixdat['Nframes'][0]
pix = pixdat['Nframes'][0]

allframes = np.empty( (framecount, height, width), dtype = np.uint16)

# for frameno, pixels in enumerate(pixdat['pixel']):
    
# p16 = pixels.astype(np.uint16).reshape(-1, 5).T

# p = allframes[frameno].reshape(-1, 4).T

# p[0, : ] = left_shift(            p16[0, :],              2) + right_shift(p16[1, :], 6)
# p[1, : ] = left_shift(bitwise_and(p16[1, :], 0b00111111), 4) + right_shift(p16[2, :], 4)
# p[2, : ] = left_shift(bitwise_and(p16[2, :], 0b00001111), 6) + right_shift(p16[3, :], 2)
# p[3, : ] = left_shift(bitwise_and(p16[3, :], 0b00000011), 8) + p16[4, :]

# <codecell>

print pixdat.dtype
print pixdat['Nframes'].shape
print pix.dtype
print pix.shape

# <codecell>

for frameno, pixels in enumerate(pix['pixel']):
    pass

# <codecell>

%%timeit 
p16 = pixels.astype(np.uint16).reshape(-1, 5).T

# <codecell>

%%timeit
for i in range(len(pixels)):
    np.uint16(pixels[i])

# <codecell>

from numba import double
from numba.decorators import jit, autojit

# <codecell>

def cast(pixels):
    p = np.empty_like(pixels, dtype = np.uint16)
    for i in range(len(pixels)):
        p[i] = np.uint16(pixels[i])

# <codecell>

cast_numba = autojit(cast)

# <codecell>

%timeit cast_numba(pixels)

# <codecell>

%%timeit
p = allframes[0].reshape(-1, 4).T

# <codecell>

p16 = pixels.astype(np.uint16).reshape(-1, 5).T
p = allframes[0].reshape(-1, 4).T
print p16.shape
print p.shape

# <codecell>

%%timeit
p[0, :] = p16[0, :]
p[0, :] *= 4
p[0, :] += np.right_shift(p16[1, :], 6)

# <codecell>

len(p[0, :])

# <codecell>

%%timeit
for i in range(len(p[0, :])):
    p[0, i] = p16[0, i]
    p[0, i] *= 4
    p[0, i] += p16[1, i] >> 6

# <codecell>

%%timeit
p[1, :] = p16[1, :]
p[1, :] &= 0b00111111
p[1, :] *= 16
p[1, :] += np.right_shift(p16[2, :], 4)

# <codecell>

%%timeit
p[2, :] = p16[2, :]
p[2, :] &= 0b00001111
p[2, :] *= 64
p[2, :] += np.right_shift(p16[3, :], 2)

# <codecell>

e0.apply_sync(lambda: globals().keys())

# <codecell>

try:
    del c
except:
    pass

# <codecell>

ip_juergenstorage = "10.210.144.61"

# <codecell>

# c = Client(profile = 'parallel')
c = Client(url_file = '/home/neubauer/Downloads/ipcontroller-client.json', timeout = 20, debug = False,
           sshserver = "%s@%s" % (username, ip_juergenstorage),
           password = password
           )

# <codecell>

c.ids

# <codecell>

dv = c[:]
dv.block = False
print dv.targets
dv.activate("all")

# <codecell>

e0 = c[0]
e0.block = False
print e0.targets
e0.activate("0")

# <codecell>

e1 = c[1]
e1.block = False
print e1.targets
e1.activate("1")

# <codecell>

sorted( c[:].apply_sync(lambda : globals().keys())[0] )

# <codecell>

import sys
sys.path.append('/extra/InVivoDog/python/cine/tools')
import cine, dogdata

# <codecell>

# only valid on juergenstorage

initialdir = "/tank/Dinesh_InVivoDog_Stimulation/Grant_2011/InVivoDog_2013_12_18/left SLN/range finding 400Hz PW 10us/"
cinefilename = "left SLN range finding 400Hz PW 10us_952_Wed Dec 18 2013 17 16 55.516 411.001.cine"

# trigger the filemenu GUI -- SLOW!
# cinefilename = None

# <codecell>

# push the names to all the engines
dv['initialdir'] = initialdir
dv['cinefilename'] = cinefilename

# <codecell>

%%px0
allcines = [os.path.basename(item) for item in glob.glob(os.path.join(initialdir, '*.cine'))]

allcines = sorted(allcines, key = None) # eventually sort cines by the date stamp, lexically sorted is WRONG

randomcine = np.random.randint(0, high = len(allcines))
print 'avoid caching: randomcine: ', randomcine

c = cine.Cine(initialdir = initialdir, cinefilename = allcines[randomcine], debug = True)

# <codecell>

%pxresult0

# <codecell>

allcines = e0['allcines']

# <codecell>

allcines

# <codecell>

r = e0.execute("c.getallframes()", silent = False, block = False)

r.wait_interactive()

# <codecell>

r.display_outputs

# <codecell>

# for local testing: use a local cluster

r = e0.execute("c = cine.Cine(initialdir = '/extra/InVivoDog/InVivoDog_2012_02_22/asymmetric SLN/MidRLN/videos', debug = True)")

# <codecell>

r.result_dict

# <codecell>

r.display_outputs()

# <codecell>

r.metadata

# <codecell>

help e0.clear

# <codecell>

# needs cine module imported in the local context
# otherwise the pickled object from the engines can't be unpickled
e0['c']

# <codecell>

# e0.clear()

# <codecell>

e0['c'].__dict__

# <codecell>

%%px0
initialdir = c.initialdir
fullcinefilename = c.cinefilename
cinefilename = os.path.split(fullcinefilename)[1]

Nframes = c.imagecount

print initialdir
print fullcinefilename
print cinefilename

print Nframes

# <codecell>

%pxresult0

# <codecell>

initialdir = e0['initialdir']
fullcinefilename = e0['fullcinefilename']
cinefilename = e0['cinefilename']
Nframes = e0['Nframes']

# <codecell>

# prepare the rest of the engines for working with the cine file
r = c[1:].execute("c = cine.Cine(initialdir = '{}', cinefilename = '{}', debug = True)".format(initialdir, 
                                                                                               cinefilename),
                  silent = False)

# <codecell>

# should be empty
r.result_dict

# <codecell>

# should be empty
r.display_outputs()

# <codecell>

import numpy as np
import os

# <codecell>

startlist = np.round(np.linspace(0, Nframes, num = len(c.ids), endpoint = False)).astype(np.int)

endlist = startlist[1:] - 1
endlist = np.hstack([endlist, [4501]])

print zip(startlist, endlist)

pids = dv.apply_sync(os.getpid)

# <codecell>

dv['framemap'] = dict(zip(pids, zip(startlist, endlist)))
dv['framemap'][0]

# <codecell>

def getNframes():
    pid = os.getpid()
    start, end = framemap[pid]
    
    print "start: {}, end: {}".format(start, end)
    t0 = time.time()
    
    frames = c.getNframes(firstframe = start, lastframe = end)
    
    t1 = time.time()
    print t0
    print t1
    print
    return frames

# distribute the function to the engines
dv['getNframes'] = getNframes

# <codecell>

r = dv.execute('Nframes = getNframes()', silent = True, block = False)

r.wait_interactive()

# <codecell>

print r.serial_time
print r.wall_time

# <codecell>

r.display_outputs()

# <codecell>

r = dv.apply_async(lambda : diffNframes.shape)

# <codecell>

r.result

# <codecell>

np.array(r.result)[:, 0].sum()

# <codecell>

diffframecounts = [item[0] for item in r.result]

# <codecell>

np.diff(np.arange(5, 0, -1, dtype = np.uint16))

# <codecell>

%%pxall
# work with differences of consecutive frames
# prevent overflow of uint16 when calculating differences
t0 = time.time()
diffNframes = np.diff(Nframes.astype(np.float), n = 1, axis = 0)
print "time for diff [s]: ", time.time() - t0

t0 = time.time()
meankymo = np.mean(np.mean(diffNframes.astype(np.float), axis = 2), axis = 0)
print "time for mean [s]: ", time.time() - t0

t0 = time.time()
m2kymo =  np.mean( np.mean( diffNframes.astype(np.float)**2, axis = 2), axis = 0)
print "time for mean-square [s]: ", time.time() - t0

# <codecell>

%pxresultall

# <codecell>

meankymos = dv.pull('meankymo', block = False)
m2kymos = dv.pull('m2kymo', block = False)

# <codecell>

meankymos = np.array(meankymos)
m2kymos = np.array(m2kymos)

# <codecell>

allmean = np.sum(meankymos * np.array(diffframecounts, dtype = np.float).reshape(-1, 1), axis = 0) / sum(diffframecounts)

# <codecell>

allm2 = np.sum(m2kymos * np.array(diffframecounts, dtype = np.float).reshape(-1, 1), axis = 0) / sum(diffframecounts)

# <codecell>

allstd = np.sqrt( allm2 - allmean**2 )

# <codecell>

%matplotlib inline
import matplotlib.pyplot as plt

# <codecell>

plt.plot(allstd)

# <codecell>

np.argmax(allstd)

# <codecell>

linenr = 256
# linenr = 400

r = dv.execute("Nkymo = Nframes[:, %d, :]" % linenr, block = False, silent = False)

r.wait_interactive()

# <codecell>

print r.wall_time
print r.serial_time

# <codecell>

r = dv.apply_async(lambda: Nkymo.shape)

# <codecell>

sum([item[0] for item in r.result])

# <codecell>

r = dv.execute("kymoshape = Nkymo.shape")

# <codecell>

import time

# <codecell>

r = dv.execute("import zlib")

# <codecell>

r = dv.execute("kymozip = zlib.compress(Nkymo.tostring(), 9)", block = False)

# <codecell>

print r.wall_time
print r.serial_time

# <codecell>

%%pxall
np.savez_compressed("/tank/temp/kymozip%d.npz" % os.getpid(), kymozip = kymozip)

# <codecell>

%pxresultall

# <codecell>

%%px0
print len(kymozip)
print len(Nkymo.tostring())

# <codecell>

%pxresult0

# <codecell>

t0 = time.time()
kymozips = dv.pull('kymozip', block = True)
dt = time.time() - t0
print "transfer time [s]: ", dt

Nbytes = sum(map(lambda x: len(x), kymozips))

print "transfer rate [Mbit/sec]: ", Nbytes / dt / 2.0**20 * 8

# <codecell>

Nbytes

# <codecell>

t0 = time.time()
Nkymos = dv.pull('Nkymo', block = True)
dt = time.time() - t0
print "transfer time [sec]: ", dt

Nbytes = sum(map(lambda x: x.nbytes, Nkymos))

print "transfer rate [Mbit/sec]: ", Nbytes / dt / 2.0**20 * 8

# <codecell>

kymolist = []
dt_engine = []

t0 = time.time()
for engine in c.ids:
    t_engine = time.time()
    kymolist.append(dv.pull('Nkymo', block = True, targets = engine))
    
    dt_engine.append(time.time() - t_engine)
    
dt = time.time() - t0

print "dt [s]: ", dt

# <codecell>

print 'transfer times in Mbit/sec for each sub-kymo:'
np.array([item.nbytes for item in kymolist]) / np.array(dt_engine) / 2.0**20 * 8

# <codecell>

r = dv.gather('Nkymo') # ('Nkymo')

r.wait_interactive()

# <codecell>

print r.wall_time
print r.serial_time

# <codecell>

kymo = r.result.T
print kymo.shape

# <codecell>

print "size in MBytes"
kymo.nbytes / 2.**20

# <codecell>

print 'transfer rate in Mbit/sec:'
kymo.nbytes / r.wall_time / 2.**20 * 8

# <codecell>

import zlib

# <codecell>

print kymo.nbytes
kymozip = zlib.compress(kymo.tostring(), 9)
print len(kymozip)

# <codecell>

e0['remotekymozip'] = kymozip

# <codecell>

%%px0
with open('/tank/temp/remotekymozip.dat', 'wb') as f:
    f.write(remotekymozip)

# <codecell>

r = e0.execute('c.getallframes()', silent = False, block = False)
r.wait_interactive()

# <codecell>

print r.serial_time
print r.wall_time

# <codecell>

r.display_outputs

# <codecell>

r = e0.execute("kymo = c.makekymo(c.allframes, linenr = 256)")
r.wait_interactive()

# <codecell>

r.display_outputs

# <codecell>

r.result_dict

# <codecell>

# kymo = e0['kymo']

# <codecell>

r = e0.pull('kymo')

# <codecell>

kymo = r.result
print kymo.shape

# <codecell>

%matplotlib?

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib as mpl
import matplotlib.pyplot as plt

# <codecell>

plt.imshow(kymo[:, :1000], aspect = None)
plt.gray()

# <codecell>

frame0 = e0.apply_sync(lambda: diffNframes[0, :, :])

# <codecell>

frame0 = e1.apply_sync(lambda: Nframes[0, :, :])

# <codecell>

plt.imshow(frame0)
plt.colorbar()
plt.gray()

# <codecell>

rc = e0['c']

# <codecell>


# <codecell>

rc.makekymo(rc.allframes, linenr = 256)

# <codecell>

np.memmap(dtype = , mode = 'r')

