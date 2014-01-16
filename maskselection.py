
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# see lasso_demo.py
# /extra/usr/share/doc/packages/python-matplotlib/examples/event_handling/lasso_demo.py
# /extra/usr/share/doc/packages/python-matplotlib/examples/widgets/lasso_selector_demo.py
# http://matplotlib.org/examples/widgets/lasso_selector_demo.html
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

outfilename = "dataexchange.npz"

data = np.load(outfilename)

calib = data['calib'].tolist()
kp_coord = data['kp_coord'].tolist()

def onselect(vertices):
    path = Path( vertices, closed = True)
    
    # fraction of a pixel difference below which vertices will be simplified out
    path.simplify_threshold = 1.0
    path.should_simplify = True
    
    # path.contains_points(points = )

    vertfilename = "maskselection.npz"
    
    savez_dict = dict(vertices = vertices)
    
    np.savez_compressed(vertfilename, **savez_dict)
   
for showside in ['left']: # , 'right']:
    kp_x = [kp[0] for kp in kp_coord[showside]]
    kp_y = [kp[1] for kp in kp_coord[showside]]
    
    fig, ax = plt.subplots(figsize = (12, 12))
    fig.suptitle('calibration: %s' % showside)
    fig.canvas.manager.set_window_title('Press any key to finish mask selection')
    
    ax.imshow( calib[showside], cmap = mpl.cm.gray, zorder = 1)
    
    # ax.plot(kp_x, kp_y, 'ro', mfc = 'None', mec = 'red', ms = 20, zorder = 10)
    
    # plt.grid(True)
    
    # plt.xlim(xmin = 0) #, xmax = 325)
    # plt.ylim(ymax = 0) #, ymax = 375)

    lasso = LassoSelector(ax = ax, onselect = onselect, useblit = True,
                          lineprops = dict(color = 'yellow', linewidth = 7, linestyle = 'dashed', marker = 'None'))
    
    print "Press any key to finish mask selection"
    while True:
        buttonpressed = plt.waitforbuttonpress(timeout = -1)
        if buttonpressed:
            break

lasso.disconnect_events()
fig.canvas.manager.set_window_title('Mask selection DONE')

plt.show()