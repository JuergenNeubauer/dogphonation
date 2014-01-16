# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import scipy as sci

# <codecell>

from scipy.stats import distributions
from scipy.optimize import curve_fit, leastsq

# <codecell>

gamma = distributions.gamma

# <codecell>

## Gamma (Use MATLAB and MATHEMATICA (b=theta=scale, a=alpha=shape) definition)

## gamma(a, loc, scale)  with a an integer is the Erlang distribution
## gamma(1, loc, scale)  is the Exponential distribution
## gamma(df/2, 0, 2) is the chi2 distribution with df degrees of freedom.

def gamma_distribution(x, a, l, loc = 0.0):
    """
    return the gamma distribution value for x with parameter a and l = lambda
    a = alpha = shape
    b = theta = scale = 1 / lambda = 1 / l
    """
    scale = 1.0 / l
    
    return gamma.pdf(x, a, loc = loc, scale = scale)

# <codecell>

g = gamma_distribution(np.arange(0, 10, 0.1), 2.0, 1.0 / 2.0)

# <codecell>

import matplotlib.pyplot as plt

# <codecell>

plt.plot(g)

# <codecell>

plt.show

# <codecell>


