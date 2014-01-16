
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

outfilename = "dataexchange.npz"

data = np.load(outfilename)

calib = data['calib'].tolist()
kp_coord = data['kp_coord'].tolist()

def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)

    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x: %1.4f, y: %1.4f, z: %1.4f'%(x, y, z)
    else:
        return 'x: %1.4f, y: %1.4f'%(x, y)

clickpoints = dict()
clicked_kp = dict()

for showside in ['left', 'right']:
    kp_x = [kp[0] for kp in kp_coord[showside]]
    kp_y = [kp[1] for kp in kp_coord[showside]]
    
    fig, ax = plt.subplots(figsize = (12, 12))
    fig.suptitle('calibration: %s\np10, p1X, p1Y, p20, p2X, p2Y' % showside)
    fig.canvas.manager.set_window_title('calibration: %s' % showside)
    
    ax.imshow( calib[showside], cmap = mpl.cm.gray, zorder = 1)
    
    ax.plot(kp_x, kp_y, 'ro', mfc = 'None', mec = 'red', ms = 20, zorder = 10)
    ax.plot(kp_x, kp_y, 'gx', ms = 7, zorder = 20)
    
    plt.grid(True)
    
    plt.xlim(xmin = 0) #, xmax = 325)
    plt.ylim(ymin = 0) #, ymax = 375)
    
    X = calib[showside]
    
    numrows, numcols = X.shape
    
    ax.format_coord = format_coord
    
    clickpoints[showside] = fig.ginput(n = 6, timeout = -1, show_clicks = True)
    
    kp_all = np.array([kp_x, kp_y])
    
    # find the keypoints closest to the clicked points
    kp_indices = []
    for pt in clickpoints[showside]:
        dx = pt[0] - kp_all[0]
        dy = pt[1] - kp_all[1]
        
        ds = np.hypot(dx, dy)
        
        kp_indices.append(np.argmin(ds))
    
    clicked_kp[showside] = kp_all[:, kp_indices]
    
    ax.plot(clicked_kp[showside][0, :], clicked_kp[showside][1, :], 'yo', mfc = 'None', mec = 'yellow', ms = 20, zorder = 20)
    
    # print the clickpoints so that the calling routine can parse the stdout of this function to get the clickpoints
    ### Will be done from the calling routine!!!
    # print "clickpoints: ", clickpoints
    
    p10, p1X, p1Y, p20, p2X, p2Y = clicked_kp[showside].T # [np.array(p).T for p in clickpoints]
    
    # direction vectors of x- and y-axes for the first set of points on plane 1 (closest to the hypothenuse face of the prism)
    e1_X = p1X - p10
    e1_Y = p1Y - p10
    
    # direction vectors for the second set of points on plane 2 of the calibration target
    e2_X = p2X - p20
    e2_Y = p2Y - p20
    
    # to display the direction show the directions a times its length
    a = 4.0
    
    l1_X = np.array([p10, p10 + a * e1_X]).T
    l1_Y = np.array([p10, p10 + a * e1_Y]).T
    
    l2_X = np.array([p20, p20 + a * e2_X]).T
    l2_Y = np.array([p20, p20 + a * e2_Y]).T
    
    ax.plot(l1_X[0, :], l1_X[1, :], 'g.-', zorder = 1000)
    ax.plot(l1_Y[0, :], l1_Y[1, :], 'b.-', zorder = 1000)
    
    ax.plot(l2_X[0, :], l2_X[1, :], 'y.-', zorder = 1000)
    ax.plot(l2_Y[0, :], l2_Y[1, :], 'm.-', zorder = 1000)
    
    fig.canvas.draw()
    
# start the event loop so the plots don't disappear and the program waits, blocks
plt.show()