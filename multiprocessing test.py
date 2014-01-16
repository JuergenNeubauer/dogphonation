# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from multiprocessing import Pool, cpu_count

pool = Pool(processes = cpu_count())

# <codecell>

def f(xy):
    x, y = xy
    return x*x, y*y*y

# <codecell>

r = pool.map(f, zip(range(10), range(10)))

# <codecell>


