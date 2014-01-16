plotvar = 'F0'

def make_RLN_SLN_plot():

    colorbarlabel = label_dict[plotvar]

    for leftRLNnum in range(Nlevels):
        # fig, ax = plt.subplots(figsize = (16, 12))
        fig, ax = plt.subplots()
        ax.set_title('left RLN %d' % leftRLNnum)

        axim = ax.imshow(var_plot[plotvar][leftRLNnum, :, :])

        axim.set_clim(np.nanmin(var_plot[plotvar]), np.nanmax(var_plot[plotvar]))
        ax.relim()

        co = ax.contour(axim.get_array(), 
                        6, 
                        colors = 'w')
        ax.clabel(co, fmt = '%.0f', fontsize = 10, inline = True)

        ax.axis([-0.5, 4.5, -0.5, 7.5])

        ax.set_ylabel('right RLN')

        ax.grid(False)

        if leftRLNnum == 0:
            cb = fig.colorbar(axim, ax = ax)
            cb.set_label(colorbarlabel)

        if leftRLNnum == Nlevels - 1:
            ax.set_xlabel('SLN')

        if False:
            plt.savefig('.pdf', orientation = 'landscape',
                        papertype = 'letter', format = 'pdf',
                        bbox_inches = 'tight', pad_inches = 0.1)
    plt.show()
