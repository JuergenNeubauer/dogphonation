# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from scipy.integrate import odeint, ode

import numpy as np

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib.pyplot as plt

plt.rc('figure', figsize = (12, 8))
plt.rc('axes', lw = 2.0)

# <codecell>

def ps_t(t, t0 = 0, t1 = 1, ps0 = 0, ps1 = 1):
    """
    linear pressure ramp, 'pressure' is just a name for the main contol parameter

    alternative: add differential form of time dependence of control parameter to model equations

    ps' = const = (ps1 - ps0) / np.float(t1 - t0)
    ps(t = 0) = ps0
    """
    return (ps1 - ps0) / np.float(t1 - t0) * (t - t0) + ps0

# <codecell>

std_r = 0.0
std_phi = 1.0e-3
std_ps = 0.0

# <codecell>

def noise():
    """
    additive random Gaussian noise, mean = 0, std = sigma
    """
    if std_r > 0:
        n_r = np.random.normal(scale = std_r)
    else:
        n_r = 0

    n_phi = np.random.normal(scale = std_phi)
    
    if std_ps > 0:
        n_ps = np.random.normal(scale = std_ps)
    else:
        n_ps = 0

    return [n_r, n_phi, n_ps]

# <codecell>

# oscillation frequency is the time scale
F0 = 100 # in Hz
omega = 2.0 * np.pi * F0

# factor for nonlinear phase growth term
b = 1

# <codecell>

# time step in units of the time scaling
Nsteps_per_period = 100
dt = 1.0 / np.float(F0) / np.float(Nsteps_per_period)

# <codecell>

# start time
t0 = 0.0
# end time
t1 = 1.0

# <codecell>

# sub- and super-critical Hopf bifurcation occurs for mu = 0
ps0 = -1.0
ps1 = 50.0

# saddle-node point of the bifurcation parameter
ps_SN = -0.25

# <codecell>

# slope of linearly increasing control parameter function
slope_ps = (ps1 - ps0) / np.float(t1 - t0)
# offset
b_ps = ps0

print "slope of subglottal pressure rise: {} Pa / sec".format(a_ps)

# <codecell>

# supercritical Hopf bifurcation, right-hand side

def supercritical(t, y, omega, b, slope_ps):
    r, phi, ps = y
    
    drdt = ps * r - r**3

    dphidt = omega + b * r**3

    dpsdt = slope_ps

    return [drdt, dphidt, dpsdt]

def supercritical_jac(t, y, omega, b, slope_ps):
    r, phi, ps = y
    
    drdr = ps - 3.0 * r**2
    drdphi = 0
    drdps = r
    
    dphidr = b * 3.0 * r**2
    dphidphi = 0
    dphidps = 0
    
    dpsdr = 0
    dpsdphi = 0
    dpsdps = 0
    
    return [[drdr, drdphi, drdps],
            [dphidr, dphidphi, dphidps],
            [dpsdr, dpsdphi, dpsdps]]

# <codecell>

# subcritical Hopf bifurcation

def subcritical(t, y, omega, b, slope_ps):
    r, phi, ps = y

    drdt = ps * r + r**3 - r**5

    dphidt = omega + b * r**3

    dpsdt = slope_ps

    return [drdt, dphidt, dpsdt]

def subcritical_jac(t, y, omega, b, slope_ps):
    r, phi, ps = y
    
    drdr = ps + 3.0 * r**2 - 5.0 * r**4
    drdphi = 0
    drdps = r
    
    dphidr = b * 3.0 * r**2
    dphidphi = 0
    dphidps = 0

    dpsdr = 0
    dpsdphi = 0
    dpsdps = 0
    
    return [[drdr, drdphi, drdps],
            [dphidr, dphidphi, dphidps],
            [dpsdr, dpsdphi, dpsdps]]

# <codecell>

Hopf = ode(subcritical, jac = subcritical_jac)

Hopf.set_integrator("vode", method = 'bdf', with_jacobian = True)

def solout(t, y):
    """
    Set callable to be called at every successful integration step

    solout should return -1 to stop integration
    otherwise, it should return None or 0
    """
    stop = False

    if stop:
        return -1
    else:
        return 0
    
# Hopf.set_solout(solout)

# <codecell>

help Hopf.f

# <codecell>

help Hopf.jac

# <codecell>

def run_iteration(rhs, jac, y0, t0, omega, b, slope_ps):
    """
    rhs: function handle for right hand side, e.g. subcritical or supercritical
    jac: Jacobian
    """
    Hopf.f = rhs
    Hopf.jac = jac
    
    Hopf.set_f_params(omega, b, slope_ps)
    
    Hopf.set_jac_params(omega, b, slope_ps)
    
    Hopf.set_initial_value(y0, t0)
    
    time = []
    results = []
    
    while Hopf.successful() and Hopf.t <= t1:
        Hopf.integrate(Hopf.t + dt, step = 0, relax = 0)

        time.append(Hopf.t)
        results.append(Hopf.y)

    r, phi, ps = np.array(results).T
    
    return time, r, phi, ps

# <codecell>

# initial conditions
r0 = 0.01
phi0 = 0
y0 = [r0, phi0, ps0]

# <codecell>

time, r, phi, ps = run_iteration(supercritical, supercritical_jac, [r0 * 100.0, phi0, ps0], t0, omega, b, slope_ps = 50)

psub = r * np.sin(phi)

omega, b, slope_ps = Hopf.f_params

# instantaneous frequency

F = (omega + b * r**3) / (2.0 * np.pi)

# <codecell>

%config InlineBackend.close_figures = False

# <codecell>

# plt.clf()

plt.plot(ps, F, '-', label = 'r0 = %.2f' % r[0])

plt.xlim(xmin = min(ps), xmax = max(ps))

plt.xlabel('ps')
plt.ylabel('frequency [Hz]')

plt.ylim(ymin = F0 - 1)

plt.title('Instantaneous frequency')

plt.legend(loc = 'upper left')

# <codecell>

# plt.clf()
# fig, ax1 = plt.subplots()

ax1.plot(time, psub, '-', label = 'r0 = %.2f' % r[0])
ax1.set_yscale('linear')

ax2 = plt.twinx(ax1)

ax2.plot(time, ps, 'r-')
ax2.set_ylabel('ps', color = 'red')

ax1.set_xlabel('time')

ax1.set_ylabel('psub')

ax1.set_xlim(xmax = 1)
ax1.grid(False, axis = 'y')

ax2.set_ylim(ymin = min(ps), ymax = max(ps))

ax1.legend(loc = 'upper left')

# <codecell>

# plt.clf()
# fig, ax1 = plt.subplots()

ax1.plot(time, r, '-', label = 'r0 = %.2f' % r[0])
ax1.set_yscale('log')
# ax1.set_xscale('log')

ax2 = plt.twiny(ax1)

ax1.set_xlim(xmax = 1)
ax1.grid(False, axis = 'both')

ax2.set_xlim(xmin = min(ps), xmax = max(ps))
ax2.grid(False, axis = 'y')

ax1.set_xlabel('time')
ax2.set_xlabel('ps')

ax1.set_ylabel('amplitude of vibration')

ax1.legend(loc = 'upper left')

ax1.set_ylim(ymin = 0.005, ymax = 10)

# <codecell>


