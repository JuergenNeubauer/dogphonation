import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import onsetplots as o

##################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

c = ['k', 'r', 'g', 'b', 'm', 'c', 'y']

for TAnum, TAcond in enumerate(o.TAconditions):
    ax.plot(o.rstrain_plot[TAcond].ravel(), 
            o.dVP_plot[TAcond].ravel(), 
            o.F0_plot[TAcond].ravel(), 'x%s' % c[TAnum], label = TAcond)

plt.xlabel('right strain [%]')
plt.ylabel('adduction dVP [%]')
ax.set_zlabel('onset frequency F0 [Hz]')

plt.legend()
##################################################
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection = '3d')

for TAnum, TAcond in enumerate(o.TAconditions):
    ax1.plot(o.rstrain_plot[TAcond].ravel(), 
             o.dVP_plot[TAcond].ravel(), 
             o.ps_plot[TAcond].ravel(), 'x%s' % c[TAnum], label = TAcond)

plt.xlabel('right strain [%]')
plt.ylabel('adduction dVP [%]')
ax1.set_zlabel('onset pressure ps [Pa]')

plt.legend()
##################################################
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection = '3d')

for TAnum, TAcond in enumerate(o.TAconditions):
    ax2.plot(o.rstrain_plot[TAcond].ravel(), 
             o.ps_plot[TAcond].ravel(), 
             o.F0_plot[TAcond].ravel(), 'x%s' % c[TAnum], label = TAcond)

plt.xlabel('right strain [%]')
plt.ylabel('onset pressure [Pa]')
ax2.set_zlabel('onset frequency [Hz]')

plt.legend()
##################################################
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection = '3d')

for TAnum, TAcond in enumerate(o.TAconditions):
    ax3.plot(o.rstrain_plot[TAcond].ravel(), 
             o.A_plot[TAcond].ravel(), 
             o.F0_plot[TAcond].ravel(), 'x%s' % c[TAnum], label = TAcond)

plt.xlabel('right strain [%]')
plt.ylabel('area [a.u.]')
ax3.set_zlabel('onset frequency [Hz]')

plt.legend()
##################################################
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection = '3d')

for TAnum, TAcond in enumerate(o.TAconditions):
    ax4.plot(o.rstrain_plot[TAcond].ravel(), 
             o.dVP_plot[TAcond].ravel(), 
             o.A_plot[TAcond].ravel(), 'x%s' % c[TAnum], label = TAcond)

plt.xlabel('right strain [%]')
plt.ylabel('adduction dVP [%]')
ax4.set_zlabel('area [a.u.]')

plt.legend()
