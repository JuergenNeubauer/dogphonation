# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cmp

# <codecell>

fig = plt.figure()

# <codecell>

s = fig.gca().imshow(np.random.normal(size = 9).reshape(3, -1))

# <codecell>

mpl.path.Path(

