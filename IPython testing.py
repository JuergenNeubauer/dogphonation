# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# from youtube_dl.extractor import soundcloud

# <rawcell>

# from IPython.display import HTML
# h = HTML("""<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=http%3A%2F%2Fapi.soundcloud.com%2Ftracks%2F94543639"></iframe>""")

# <rawcell>

# h

# <codecell>

import matplotlib as mpl
mpl.get_backend()

# <codecell>

mpl.use('module://IPython.kernel.zmq.pylab.backend_inline')
mpl.get_backend()

# <codecell>

from matplotlib import pyplot as plt
plt.get_backend()

# <codecell>

%matplotlib inline

# <codecell>

%config

# <codecell>

%config InlineBackend

# <codecell>

%config InlineBackend.close_figures = True #False

# <codecell>

print mpl.is_interactive()
mpl.interactive(False)
print mpl.is_interactive()

# <codecell>

print plt.isinteractive()
plt.interactive(False)
plt.ioff()
print plt.isinteractive()

# <codecell>


# <codecell>

fig, ax = plt.subplots(nrows = 2, ncols = 2, squeeze = True)

# plt.show()

# <codecell>

fig

# <codecell>

plt.close('all')

# <codecell>

plt.show()

# <codecell>

ax

# <codecell>

ax1 = ax[0, 0]

# <codecell>

ax1

# <codecell>

ax1.axes

# <codecell>

ax1.get_adjustable()

# <codecell>

ax1.get_position()

# <codecell>

bbox = mpl.transforms.Bbox.from_bounds(0, 0, 0.2, 0.4)

# <codecell>

bbox

# <codecell>

ax1.set_position(bbox, which = 'both')

# <codecell>

ax1.get_position()

# <codecell>

ax1.get_zorder()

# <codecell>

ax1.set_zorder(10)

# <codecell>

fig

# <codecell>


# <codecell>

from IPython.display import display_png, display

# <codecell>

help display_png

# <codecell>

help display

# <codecell>

display(fig)

# <codecell>


