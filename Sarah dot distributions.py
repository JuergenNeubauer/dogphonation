# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

Nwarriors = 1024

wx = np.random.uniform(low = -1, high = 1, size = Nwarriors)
wy = np.random.uniform(low = -1, high = 1, size = Nwarriors)

# <codecell>

# background grid
bx = np.linspace(1, 4 * np.sqrt(Nwarriors), num = np.sqrt(Nwarriors))
by = np.linspace(1, 4 * np.sqrt(Nwarriors), num = np.sqrt(Nwarriors))

bx = np.repeat(bx.reshape(-1, 1), np.sqrt(Nwarriors), axis = 1)
by = np.repeat(by.reshape(-1, 1), np.sqrt(Nwarriors), axis = 1).T

# <codecell>

bx.size

# <codecell>

plt.close('all')

h = 20
w = 16 / 9. * h

plt.figure(figsize = (w, h))

plt.plot(bx, by, 'r.')

plt.xlim(xmin = 0)
plt.ylim(ymin = 0)

ax = plt.gca()
ax.set_frame_on(False)

plt.show()

# <codecell>

px = bx + wx.reshape(-1, np.sqrt(Nwarriors))
py = by + wy.reshape(-1, np.sqrt(Nwarriors))

px = px.ravel()
py = py.ravel()

# <codecell>

px.max

# <codecell>

from matplotlib.path import Path
import matplotlib.patches as patches

def stickfigure(leftfoot = (0, 0), linewidth = 5, color = 'red'):
    """
    return: body, head
    """
    template = [
                (0, 0), # left leg
                (1, 1), # hips
                (2, 0), # right leg
                (1, 1), # hips
                (1, 2.5), # neck
                (0, 1.5), # left arm
                (1, 2.5), # neck
                (2, 1.5), # right arm
                ]
    
    verts = np.array(template) + np.array(leftfoot)
    
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.MOVETO,
             Path.LINETO
             ]
    
    path = Path(verts, codes)

    body = patches.PathPatch(path, facecolor='none', edgecolor = color, linewidth = linewidth)
    
    headpos = np.array([1, 3]) + np.array(leftfoot)
    head = plt.Circle(headpos.tolist(), radius = 0.5, edgecolor = color, facecolor = color, linewidth = linewidth)

    return body, head

# <codecell>

plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(111)

body, head = stickfigure(leftfoot = (0, 0))

ax.add_patch(body)
ax.add_artist(head)

ax.set_xlim(0, 3)
ax.set_ylim(0, 4)

ax.set_aspect('equal')

plt.show()

# <codecell>

xmin = px.min()
xmax = px.max()
xrange = xmax - xmin
ymin = py.min()
ymax = py.max()
yrange = ymax - ymin

kingroup = np.where(px > xmin + 0.2 * xrange, 0, 1)
kingroup = np.where(px < xmin + 0.4 * xrange, kingroup, 1)

kingroup = np.where(py > ymin + 0.2 * yrange, kingroup, 1)
kingroup = np.where(py < ymin + 0.4 * yrange, kingroup, 1)

# <codecell>

kinindex = np.where(kingroup == 0)[0]
print 'number of people in region to choose from: ', len(kinindex)

# choose 10 from the sub-selection which was regionally constraint
kinindex = np.random.choice(kinindex, size = 10, replace = False)

print 'kin index list: ', kinindex

# <codecell>

colors = ['white'] * Nwarriors

for ind in kinindex:
    colors[ind] = 'red'

# <codecell>

plt.close('all')

h = 20
w = 16 / 9. * h

fig, ax = plt.subplots(figsize = (w, h), frameon = False, facecolor = 'black', edgecolor = 'black')
ax.set_frame_on(False)

fig.set_frameon(True)
fig.set_facecolor('black')
fig.set_edgecolor('black')

if False:
    ax.plot(px, py, 
            'o', 
            mec = 'white', mfc = 'white', alpha = 1.0)

for x, y, c in zip(px, py, colors):
    # c = 'white'
    body, head = stickfigure(leftfoot = (x, y), linewidth = 3, color = c)
    
    ax.add_patch(body)
    ax.add_artist(head)
    
plt.xticks([])
plt.yticks([])

plt.xlabel('')
plt.ylabel('')

plt.xlim(xmin = -1.5, xmax = px.max() + 4)
plt.ylim(ymin = -1.5, ymax = py.max() + 5)

plt.savefig('kin1000people.png', bbox_inches = 'tight', facecolor = 'black', edgecolor = 'black')

plt.show()

# <codecell>

plt.close('all')

fig, ax = plt.subplots(figsize = (15, 15))

centers = [(20, 20), (50, 25), (23, 55)]
radii = [15, 14, 16]
colors = ['k', 'r', 'g', 'b', 'orange', 'magenta', 'cyan', 'yellow', 'blue', 'pink']

for center, radius, color in zip(centers, radii, colors):
    c = plt.Circle(center, radius = radius, ec = color, fc = 'None', lw = 10)

    ax.add_artist(c)

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.show()

# <codecell>


