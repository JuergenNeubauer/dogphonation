# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import zmq
import numpy as np

# <codecell>

from multiprocessing import Process

# <codecell>

context = zmq.Context()

# <codecell>

dealer = context.socket(zmq.DEALER)
dealer1 = context.socket(zmq.DEALER)

router = context.socket(zmq.ROUTER)

# <codecell>

dealer.IDENTITY = 'dealer' # worker for dogdata
dealer1.IDENTITY = 'dealer1' # worker for cine

router.IDENTITY = 'router' # client application code 
router.ROUTER_MANDATORY = True

# <codecell>

router.bind("inproc://testing")

dealer.connect("inproc://testing")
dealer1.connect("inproc://testing")

# <codecell>

A = np.ones(5, dtype = np.uint8)

# <codecell>

A *= 2

# <codecell>

message = ['5', '3', A]
# message = A

# <codecell>

message[0] = 'a'

# <codecell>

router.send_multipart(['dealer'] + message, copy = False)

# <codecell>

msg = dealer.recv_multipart(flags = zmq.NOBLOCK, copy = False)

# <codecell>

msg

# <codecell>

for frame in msg:
    print frame.bytes
    try:
        print np.frombuffer(buffer(frame), dtype = np.uint8)
    except Exception as e:
        print e

# <codecell>

dealer.send_multipart(message, copy = False, track = False)

# <codecell>

dealer1.send_multipart(message, copy = False, track = False)

# <codecell>

try:
    msg_parts = router.recv_multipart(flags = zmq.NOBLOCK, copy = False, track = True)
except Exception as e:
    print e

# <codecell>

len msg_parts

# <codecell>

for frame in msg_parts:
    print frame.bytes
    print np.frombuffer(buffer(frame), dtype = np.uint8)

# <codecell>

msg_parts[0] = 'dealer'
msg_parts[1] = '6'

# <codecell>

for frame in msg_parts:
    print frame

# <codecell>

tracker = router.send_multipart(msg_parts, flags = zmq.NOBLOCK, copy = False, track = True)

# <codecell>

tracker.done

# <codecell>

msg = dealer1.recv_multipart(flags = zmq.NOBLOCK, copy = False, track = True)

# <codecell>

len msg

# <codecell>

for m in msg:
    print m.bytes

# <codecell>

import os, getpass

# <codecell>

getpass.getuser()

# <codecell>

for items in os.environ.items():
    if 'neubauer' in items[1]:
        print items   

# <codecell>

os.getlogin()

# <codecell>

os.path.expanduser('~')

# <codecell>

env = os.environ

with open('test.txt', 'wt') as f:
    f.write("{}: {}\n".format('getpass.getuser', getpass.getuser()))
    for key in env.keys():
        f.write("{}: {}\n".format(key, env[key]))

# <codecell>


