# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import requests
import os, sys, glob
import numpy as np

from IPython.display import HTML, display_html

import time

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')
import cine, cinefile_datatypes

# <codecell>

url = "http://virtual-larynx.org/InVivoDog/"

# <codecell>

# authentication is needed for HEAD and OPTIONS requests

r = requests.head(url) # , auth = ('testuser', 'passmein'))

# <codecell>

r

# <codecell>

r.headers.items()

# <codecell>

# could also use requests.options(url, ...)
# authentication is also needed for this!!!
r = requests.request('OPTIONS', url) #, auth = ('testuser', 'passmein'))

# <codecell>

r

# <codecell>

r.headers.items()

# <codecell>

# has side effect of styling the entire Notebook according to the CSS style sheet in r.content
# display_html(r.content, raw = True)

# <codecell>

s = requests.Session()
s.streaming = True

# <codecell>

s.auth = ('testuser', 'passmein')

# <codecell>

s.headers.items()

# <codecell>

s.cookies

# <codecell>

r = s.head(url = url)

# <codecell>

r

# <codecell>

r.headers.items()

# <codecell>

r = s.options(url)

# <codecell>

r

# <codecell>

r.headers.items()

# <codecell>

allprop = """<?xml version="1.0" encoding="utf-8" ?>
             <D:propfind xmlns:D="DAV:">
             <D:allprop/>
             </D:propfind>""".split('\n')

# <codecell>

allproperties = """<?xml version="1.0" encoding="utf-8" ?> 
                   <propfind xmlns="DAV:"> 
                   <propname/> 
                   </propfind>"""

# <codecell>

xmldata = reduce(lambda x, y: x + y, allprop)

# <codecell>

# xmldata = reduce(lambda x, y: x + y, allproperties)

# <codecell>

r = s.request(method = 'PROPFIND', url = url, data = xmldata, headers = dict(Depth = '1'), stream = s.streaming, verify = None)

# <codecell>

r.headers.items()

# <codecell>

r

# <codecell>

r.content.split('\n')

# <codecell>

r.url

# <codecell>

r = s.request(method = 'GET', url = url + "InVivoDog_2012_10_17/left PCA/range finding")

# <codecell>

r

# <codecell>

r.headers.items()

# <codecell>

display_html(r.content, raw = True)

# <codecell>

r = s.request(method = 'GET', 
url = url + "InVivoDog_2012_10_17/left PCA/range finding/left PCA range finding_000_Wed Oct 17 2012 14 18 19.536 539.001.cine",
stream = s.streaming)

# <codecell>

r

# <codecell>

r.headers.items()

# <codecell>

def get_cine_info(dav_response, datatype):
    r = dav_response
    
    # generator_cinefileheader = r.iter_content(chunk_size = cinefile_datatypes.cinefileheader_type.itemsize)
    # cinefileheader = generator_cinefileheader.next()

    headerdata = r.raw.read(amt = datatype.itemsize)
    
    np_data = np.fromstring(headerdata, count = 1, dtype = datatype)
     
    return dict(zip(datatype.names, np_data.item()))

# <codecell>

cinefileheader = get_cine_info(r, cinefile_datatypes.cinefileheader_type)
cinefileheader

# <codecell>

# generator_bitmapinfoheader = r.iter_content(chunk_size = cinefile_datatypes.bitmapinfoheader_type.itemsize)
# bitmapinfoheader = generator_bitmapinfoheader.next()

# bitmapinfoheader = r.raw.read(amt = cinefile_datatypes.bitmapinfoheader_type.itemsize)

bitmapinfoheader = get_cine_info(r, cinefile_datatypes.bitmapinfoheader_type)
bitmapinfoheader

# <codecell>

setup = get_cine_info(r, cinefile_datatypes.setup_type)

# <codecell>

def get_rawframe_positions(dav_response):
    r = dav_response
    
    bytes_read = reduce(lambda x, y: x + y, 
        [getattr(cinefile_datatypes, t).itemsize for t in ['cinefileheader_type', 'bitmapinfoheader_type', 'setup_type']])    
    
    skip_bytes = cinefileheader['offimageoffsets'] - bytes_read
    
    skip_data = r.raw.read(amt = skip_bytes)
    
    pImage_type = np.dtype( [ ('pImage', np.int64, cinefileheader['imagecount']) ] )
    
    pImage_raw = r.raw.read(amt = pImage_type.itemsize)
    
    pImage = np.fromstring(pImage_raw, count = 1, dtype = pImage_type)
    
    return p['pImage']

# <codecell>

pImage = get_rawframe_positions(r)

# <codecell>

# image size in bytes of each raw byte-packed frame
bisizeimage = bitmapinfoheader['bisizeimage']

frame_type = np.dtype([('annotationsize', np.uint32),
                       ('imagesize', np.uint32),
                       ('pixel', np.uint8, bisizeimage),
                       ])

# <codecell>

def get_rawframe(dav_response):
    r = dav_response
    
    frame_raw = r.raw.read(amt = frame_type.itemsize)
    
    return np.fromstring(frame_raw, count = 1, dtype = frame_type)

# <codecell>

frame = get_rawframe(r)

# <codecell>

frame['pixel'].shape

# <codecell>

height = bitmapinfoheader['biheight'] 
width = bitmapinfoheader['biwidth']

allframes = np.empty( (1, height, width), dtype = np.uint16)

allframes = cine.Cine._rawtopixel(frame, allframes)

# <codecell>

p = requests.post(url = url + "kymogram.py", auth = ('testuser', 'passmein'), 
                  data = dict(data = 'hello'), params = dict(line = '10'))

# <codecell>

p = requests.post(url = "http://httpbin.org/post", auth = ('testuser', 'passmein'), 
                  data = dict(file = 'testcine.cine'), params = dict(cine = 'test.cine', kymoline = '10'))

# <codecell>

p.encoding

# <codecell>

p.request.body

# <codecell>

p.url

# <codecell>

p.text.splitlines()

# <codecell>

p.headers.items()

# <codecell>

p.content.splitlines()

# <codecell>

requests.__version__

# <codecell>

s = requests.Session()
s.auth = ('testuser', 'passmein')

# s.data = dict(data = 'hello')
s.params = dict(cine = 'testcine.cine', 
                line = '10')

try:
    del p
except:
    pass

# <codecell>

p = s.request(method = 'GET', url = url.replace('InVivoDog/', '') + "modpythontest.py", 
              # data = s.data, 
              stream = True)

# <codecell>

p.url

# <codecell>

p.request.body

# <codecell>

p.headers.items()

# <codecell>

requests.utils.dict_from_cookiejar(p.cookies)

# <codecell>

if not p.ok:
    print p.content

# <codecell>

if int(p.headers['content-length']) < 1000:
  print p.content

# <codecell>

p.content.splitlines()

# <codecell>

for k in xrange(0):
    t0 = time.time()
    
    s.params = dict(line = str(k))
                    
    p = s.request(method = 'POST', url = url.replace('InVivoDog/', '') + "modpythontest.py", stream = True)

    print '\n'.join([item for item in p.content.splitlines() if item])
    
    print "dt1 [s]: ", time.time() - t0
    sys.stdout.flush()
    
    time.sleep(2)
    
    print "dt2 [s]: ", time.time() - t0
    sys.stdout.flush()

# <codecell>

# chunk = p.iter_content(chunk_size = 100)

# chunk = p.raw.read(amt = 200)
# chunk

# <codecell>

t0 = time.time()

if p.raw.closed:
    print "stream was closed, requesting new data\n"
    p = s.request(method = 'POST', url = url.replace('InVivoDog/', '') + "modpythontest.py",
                  params = dict(cine = 'testcine_other.cine', 
                                line = '12',
                                debug = 'no'),
                  stream = True)
else:
    print "stream still open, picking up data from stream\n"
    
print p.url
for item in p.headers.items():
    print item
print    

print requests.utils.dict_from_cookiejar(p.cookies)
print

for key in ['hits', 'cine', 'line', 'debug', 'pwd']:
    print "{}: {}".format(key, p.headers[key])

Nbytes = int(p.headers['content-length'])
print 'content-length: {} bytes'.format(Nbytes)
print

allchunks = ''
chunksize = 2**14

# for numchunk, chunk in enumerate(p.iter_content(chunk_size = chunksize)):
#     allchunks += chunk
# a = np.fromstring(allchunks, count = Nbytes/2, dtype = np.uint16)

allchunks = [chunk for chunk in p.iter_content(chunk_size = chunksize)]
numchunk = len(allchunks)
a = np.fromstring(''.join(allchunks), count = Nbytes/2, dtype = np.uint16)

dt = time.time() - t0
print "transfer time [ms]: ", dt * 1000 
print "transfer speed [MBytes / sec]: ", Nbytes / dt / 2.0**20
print "transfer speed [Mbits / sec]: ", Nbytes * 8 / dt / 2.0**20

print a[:40]

print "number uint16 numbers in transfered array: ", a.shape[0]

print "numchunk: ", numchunk

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools/')

# <codecell>

import cine
reload(cine)

# <codecell>

c = cine.Cine(initialdir = '/extra/InVivoDog/InVivoDog_2012_10_17/left PCA/range finding', cinefilename = '', debug = True)

# <codecell>

del c

# <codecell>


