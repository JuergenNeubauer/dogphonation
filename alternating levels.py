# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

Nlevels = 11

# <codecell>

startlevel = np.array([1, 0], dtype = np.uint16).reshape(1, 2)

# <codecell>

inclevel = np.array([0, 1], dtype = np.int16).reshape(1, 2)

# <codecell>

rellevels = [[0, 0, 0]]

count = 0

print rellevels[-1]

newlevel = startlevel.copy()

for level in range(1, Nlevels):
    # print 'level: ', level
    for sublevel in range(level):
        count += 1
        rellevels.append([count] + newlevel.squeeze().tolist())
        print rellevels[-1]
        
        count += 1
        rellevels.append([count] + np.fliplr(newlevel).squeeze().tolist())
        print rellevels[-1]
        
        newlevel += inclevel
    
    count += 1
    rellevels.append([count] + newlevel.squeeze().tolist())
    print rellevels[-1]

    newlevel += inclevel
    inclevel = np.fliplr(inclevel)
    for sublevel in range(level):
        newlevel -= inclevel

# <codecell>

import csv

with open('relative_levels.csv', 'wt') as f:
    writer = csv.writer(f)
    
    writer.writerow(['index', 'pattern A', 'pattern B'])
    
    for row in rellevels:
        writer.writerow(row)

# <codecell>


