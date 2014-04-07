# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# import this into other notebooks as so:

# %run -i 'tools for implant analysis.py'

# <codecell>

def isoamp_adjustment(d):
    # need to correct ps, Q, and EMG1 because of 2:1 iso-amp which also has some channel dependent offset
    Vconv = d.convEMG # EMG conversion is just conversion from numbers to Volts
    
    # WRONG: d.allps = isoampgain_ps * d.allps - Voffset_ps / Vconv * d.convps
    d.allps = (d.allps / d.convps - Voffset_ps / Vconv) * isoampgain_ps * d.convps
    
    # WRONG: d.allQ = isoampgain_Q * d.allQ - Voffset_Q / Vconv * d.convQ
    d.allQ = (d.allQ / d.convQ - Voffset_Q / Vconv) * isoampgain_Q * d.convQ

# <codecell>

## left vagal nerve paralysis
##
## relative levels from column 1 for right SLN
## and column 3 for right RLN

# num_rightSLN = 1
# num_rightRLN = 3
# 
# relative_nerve_levels = [('rightSLN', num_rightSLN), 
#                          ('rightRLN', num_rightRLN)]

##########################################################################

def getonsetdata(basedir, implant_vagal, relative_nerve_levels):
    
    nerve01_name, nerve01_ind = relative_nerve_levels[0]
    nerve02_name, nerve02_ind = relative_nerve_levels[1]
    
    print "nerve01_name: {}, nerve01_ind: {}".format(nerve01_name, nerve01_ind)
    print "nerve02_name: {}, nerve02_ind: {}".format(nerve02_name, nerve02_ind)
    print
    
    min_ps = dict(casename = None, value = np.infty)
    min_Q = dict(casename = None, value = np.infty)
    
    for casename in implant_vagal:
        
        hdf5dirname = os.path.join(basedir, implant_vagal[casename]['hdf5datadir'])
        if not os.path.isdir(hdf5dirname):
            print "no hdf5 directory found: ", hdf5dirname
            continue
    
        hdf5filename = glob.glob(os.path.join(hdf5dirname, '*.hdf5'))
        if len(hdf5filename) > 1:
            print "found more than 1 data files in dir: ", hdf5dirname
            # print "skipping for now"
            # print
            # continue
            print sorted(hdf5filename, key = datetimestamp)
            hdf5filename = sorted(hdf5filename, key = datetimestamp)[-1]
            print "selecting the latest one: ", hdf5filename
        else:
            hdf5filename = hdf5filename[-1]
            
        print casename
        
        d = dogdata.DogData(datadir = hdf5dirname, datafile = os.path.basename(hdf5filename))
    
        d.get_all_data()
        
        print "number of stimulation levels: ", d.Nlevels
        
        if np.min(d.allps) < min_ps['value']:
            min_ps['casename'] = casename
            min_ps['value'] = np.min(d.allps)
            
        if np.min(d.allQ) < min_Q['value']:
            min_Q['casename'] = casename
            min_Q['value'] = np.min(d.allQ)
        
        print "before isoamp adjustments:"
        print "min_ps: {}, max_ps: {}".format(np.min(d.allps), np.max(d.allps))
        print "min_Q: {}, max_Q: {}".format(np.min(d.allQ), np.max(d.allQ))
        print
              
        isoamp_adjustment(d)
    
        print "after isoamp adjustments:"
        print "min_ps: {}, max_ps: {}".format(np.min(d.allps), np.max(d.allps))
        print "min_Q: {}, max_Q: {}".format(np.min(d.allQ), np.max(d.allQ))
        print
    
        F0 = np.ones((d.Nlevels, d.Nlevels)) * np.nan
        onsettime_ms = np.ones_like(F0) * np.nan
        ps_onset = np.ones_like(F0) * np.nan
        Q_onset = np.ones_like(F0) * np.nan
        
        nerve01 = np.ones_like(F0) * np.nan
        nerve02 = np.ones_like(F0) * np.nan
        stimind = np.ones_like(F0) * np.nan
    
        for stimnum, (nerve01_level, nerve02_level) in enumerate(d.a_rellevels[:, [nerve01_ind, nerve02_ind]]):
            
            nerve_xaxis = nerve02_level # rightRLNlevel
            nerve_yaxis = nerve01_level # rightSLNlevel
            
            stimind[nerve_yaxis, nerve_xaxis] = stimnum
            # rightSLN[nerve_yaxis, nerve_xaxis] = rightSLNlevel
            nerve01[nerve_yaxis, nerve_xaxis] = nerve01_level
            # rightRLN[nerve_yaxis, nerve_xaxis] = rightRLNlevel
            nerve02[nerve_yaxis, nerve_xaxis] = nerve02_level

            try:
                onsettime_ms[nerve_yaxis, nerve_xaxis] = implant_vagal[casename]['onsettime_ms'][stimnum]
            except Exception as e:
                print e
                print "stimnum: ", stimnum
                print "nerve_xaxis: ", nerve_xaxis
                print "nerve_yaxis: ", nerve_yaxis
                print "onsettime_ms.shape: ", onsettime_ms.shape
            
            F0[nerve_yaxis, nerve_xaxis] = implant_vagal[casename]['F0'][stimnum]
            
            ps_onset[nerve_yaxis, nerve_xaxis] = np.interp(onsettime_ms[nerve_yaxis, nerve_xaxis] / 1000., 
                                                           d.time_psQ, d.allps[stimnum, :],
                                                           left = np.nan, right = np.nan)
            
            Q_onset[nerve_yaxis, nerve_xaxis] = np.interp(onsettime_ms[nerve_yaxis, nerve_xaxis] / 1000.,
                                                          d.time_psQ, d.allQ[stimnum, :],
                                                          left = np.nan, right = np.nan)
            
        implant_vagal[casename]['nerve_xaxis'] = nerve02_name
        implant_vagal[casename]['nerve_yaxis'] = nerve01_name
        
        implant_vagal[casename][nerve01_name] = nerve01 # rightSLN
        implant_vagal[casename][nerve02_name] = nerve02 # rightRLN
        
        implant_vagal[casename]['stimind'] = stimind
        
        implant_vagal[casename]['F0'] = F0
        implant_vagal[casename]['onsettime_ms'] = onsettime_ms
        implant_vagal[casename]['ps_onset'] = ps_onset
        implant_vagal[casename]['Q_onset'] = Q_onset
        
    print "before correction"
    print "min_ps: ", min_ps
    print "min_Q: ", min_Q
    
    return min_ps, min_Q

# <codecell>

def plotonsetdata(implant_vagal, name_paralysis = 'vagal nerve paralysis', F0_normalized = True, ps_normalized = True, Q_normalized = True):

    min_F0 = np.min([np.nanmin(implant_vagal[casename]['F0']) for casename in implant_vagal])
    max_F0 = np.max([np.nanmax(implant_vagal[casename]['F0']) for casename in implant_vagal])

    print "min_F0: {} Hz, max_F0: {} Hz".format(min_F0, max_F0)

    min_ps_onset = np.min([np.nanmin(implant_vagal[casename]['ps_onset']) for casename in implant_vagal])
    max_ps_onset = np.max([np.nanmax(implant_vagal[casename]['ps_onset']) for casename in implant_vagal])
    
    print "min_ps_onset: {}, max_ps_onset: {}".format(min_ps_onset, max_ps_onset)
    
    min_Q_onset = np.min([np.nanmin(implant_vagal[casename]['Q_onset']) for casename in implant_vagal])
    max_Q_onset = np.max([np.nanmax(implant_vagal[casename]['Q_onset']) for casename in implant_vagal])
    
    print "min_Q_onset: {}, max_Q_onset: {}".format(min_Q_onset, max_Q_onset)
    
    plt.close('all')
    
    for casename in implant_vagal:
        xlabel = implant_vagal[casename]['nerve_xaxis']
        ylabel = implant_vagal[casename]['nerve_yaxis']
        
        
        F0 = implant_vagal[casename]['F0']
        
        try:
            plt.clf()
        except:
            pass
        
        plt.imshow(F0)
        
        plt.xlabel(xlabel) # ('right RLN')
        plt.ylabel(ylabel) # ('right SLN')
        
        if F0_normalized:
            plt.clim(vmin = min_F0, vmax = max_F0)
        
        plt.title("%s: %s" % (name_paralysis, casename))
        
        cb = plt.colorbar()
        cb.set_label('frequency [Hz]')
        
        plt.grid(False)
        
        savename = "%s.F0.%s.pdf" % (name_paralysis.replace(' ', '_'), casename)
        
        if F0_normalized:
            savename = savename.replace('.F0.', '.F0.Normalized.')
        
        plt.savefig(savename, 
                    orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)
        
        ######################################################################################
        
        ps_onset = implant_vagal[casename]['ps_onset']
        
        plt.clf()
        
        plt.imshow(ps_onset)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if ps_normalized:
            plt.clim(vmin = min_ps_onset, vmax = max_ps_onset)
        
        plt.title("%s: %s" % (name_paralysis, casename))
        
        cb = plt.colorbar()
        cb.set_label('ps [Pa]')
        
        plt.grid(False)
        
        savename = '%s.Ps.%s.pdf' % (name_paralysis.replace(' ', '_'), casename)
        
        if ps_normalized:
            savename = savename.replace('.Ps.', '.Ps.Normalized.')
    
        plt.savefig(savename,
            orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)
        
        ######################################################################################
        
        Q_onset = implant_vagal[casename]['Q_onset']
        
        plt.clf()
        
        plt.imshow(Q_onset)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if Q_normalized:
            plt.clim(vmin = min_Q_onset, vmax = max_Q_onset)
            
        plt.title("%s: %s" % (name_paralysis, casename))
        
        cb = plt.colorbar()
        cb.set_label('Q [ml/s]')
        
        plt.grid(False)
        
        savename = '%s.Q.%s.pdf' % (name_paralysis.replace(' ', '_'), casename)
        
        if Q_normalized:
            savename = savename.replace('.Q.', '.Q.Normalized.')
    
        plt.savefig(savename,
            orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)        
            
        ######################################################################################
                
        # power = ps_onset * Q_onset

# <codecell>

# savevarnames = ['phase lead', 'mucosal amp', 'vibratory amp']

def plotsymmetry(TAasymmetry, symvars = ['phase lead', 'mucosal amp', 'vibratory amp'], name_paralysis = 'TA asymmetry'):

    symmetry_cmap = mpl.colors.ListedColormap(['b', 'c', 'g', 'orange', 'r'])
    symmetry_cmap.set_under(color = 'k')
    symmetry_cmap.set_over(color = 'gray')

    symmetry_cmap = mpl.colors.ListedColormap(['b', 'g', 'r'])
    
    plt.close('all')
    
    for casename in TAasymmetry:
        xlabel = TAasymmetry[casename]['nerve_xaxis']
        ylabel = TAasymmetry[casename]['nerve_yaxis']
        
        for varname in symvars:

            var = TAasymmetry[casename]['onset ' + varname]

            fig, ax = plt.subplots(# figsize = ()
                                   )
            axim = ax.imshow(var, cmap = symmetry_cmap, vmin = -1, vmax = 1)

            Nlevel = int(np.sqrt(TAasymmetry[casename]['Nstimulation']) - 1)

            ax.plot([-0.5, Nlevel + 0.5], [-0.5, Nlevel + 0.5], 'k-')

            ax.axis([-0.5, Nlevel + 0.5, -0.5, Nlevel + 0.5])

            ax.set_xticks(range(Nlevel+1))
            ax.set_yticks(range(Nlevel+1))

            for xmaj in ax.xaxis.get_majorticklocs():
                ax.axvline(x = xmaj + 0.5, ls = ':', color = 'k', lw = 1, marker = 'None')

            for ymaj in ax.yaxis.get_majorticklocs():
                ax.axhline(y = ymaj + 0.5, ls = ':', color = 'k', lw = 1, marker = 'None')
            
            plt.grid(False)
            
            plt.title("{}: {}".format(casename, varname))
            
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
            cb = plt.colorbar(axim, ax = ax) #, extend = 'both')
            cb.solids.set_edgecolor('face')
            cb.set_label(varname)
            
            cb.set_ticks([-0.6, 0, 0.6])
            cb.set_ticklabels(['left', 'symmetric', 'right'])
            
            ytl = cb.ax.get_yticklabels()
            for text in ytl:
                text.set_rotation(90)

            plt.title("%s: %s" % (name_paralysis, casename))
            
            savename = "{}.{}.{}.pdf".format(name_paralysis.replace(' ', '_'), 
                                             varname.replace(' ', '_'), 
                                             casename)
            
            plt.savefig(savename, 
                        orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

def Bernoulli_Power(implant_recurrens):
    for casename in implant_recurrens:
        ps = implant_recurrens[casename]['ps_onset']
        Q = implant_recurrens[casename]['Q_onset']
    
        A = Q / np.sqrt(ps) # Bernoulli area at onset
    
        P = ps * Q * 1e-6 # aerodynamic power at onset, originally in Pa * ml/s = Pa * cm^3 / s = 1e-6 W
        
        implant_recurrens[casename]['A_onset'] = A
        implant_recurrens[casename]['P_onset'] = P

# <codecell>

varnames = ['F0', 'ps_onset', 'Q_onset', 'A_onset', 'P_onset']
varlabels = ['onset F0 [Hz]', 'onset ps [Pa]', 'onset Q [ml/s]', 
             'onset Bernoulli area [a.u]', 'onset aerodynamic power [W]']

# <codecell>

def statistics(implant_recurrens, varname = 'F0'):
    for casename in implant_recurrens:
        no_nans = ~np.isnan(implant_recurrens[casename][varname])
        data = implant_recurrens[casename][varname][no_nans]
        
        implant_recurrens[casename][varname + '_median'] = np.median(data)
        implant_recurrens[casename][varname + '_percentiles'] = np.percentile(data, q = [25, 75])

# <codecell>

import numpy as np
from scipy import stats

# <codecell>

def plot_boxplot(implant_recurrens, casenames, varname = 'F0', label = 'F0 [Hz]', 
                 title = 'recurrent nerve paralysis'):
    
    plt.close('all')
    
    plt.figure(figsize = (20, 10))
    
    data = [implant_recurrens[casename][varname][~np.isnan(implant_recurrens[casename][varname])] 
            for casename in casenames]
    
    plt.boxplot(data)
    
    plt.xticks(np.arange(1, len(casenames)+1), casenames, rotation = 60)
    
    plt.ylabel(label)
    
    plt.grid(axis = 'x')
    
    plt.title(title)
    
    figname = '{}.{}.boxplot.pdf'.format(title.replace(' ', '_'), varname)
    plt.savefig(figname, orientation = 'landscape', bbox_inches = 'tight')

# <codecell>

def plot_statistics(implant_recurrens, casenames, varname = 'F0', label = 'F0 [Hz]', 
                    title = 'recurrent nerve paralysis'):
    
    medians = [implant_recurrens[casename][varname + '_median'] for casename in casenames]
    q1, q3 = zip( *[implant_recurrens[casename][varname + '_percentiles'] for casename in casenames] )
    
    whisker = 1.5
    # interquartile data range
    iq = q3 - q1 # data range
    
    high_val = q3 + whisker * iq
    low_val = q1 - whisker * iq
    
    # high_val - low_val = iq + 2 * whisker * iq
    
    cases = np.arange(len(casenames))
    
    plt.close('all')
    
    plt.figure(figsize = (10, 8))
    
    plt.plot(medians, cases, 'r.', ms = 20)
    
    plt.plot(q1, cases, 'b|', ms = 20)
    plt.plot(q3, cases, 'b|', ms = 20)
    
    plt.yticks(cases, casenames, 
               # rotation = 90
               )
    
    plt.xlabel(label)
    
    plt.ylim(-0.5, cases[-1] + 0.5)
    
    plt.grid(axis = 'x')
    
    plt.title(title)
    
    figname = '{}.{}.stats.pdf'.format(title.replace(' ', '_'), varname)
    plt.savefig(figname, orientation = 'landscape', bbox_inches = 'tight')

# <codecell>

def distances(clickdata):
    if clickdata is None:
        return None, None
    if clickdata is np.nan:
        return None, None
    if type(clickdata) is list:
        clickdata = np.array(clickdata)
    
    # baseline vectors between anterior landmark and VP on left and right sides
    vlr_clickdata = clickdata[1:, :] - clickdata[0, :]
    
    # number of points along vocal fold, but NOT at the anterior tip
    Nclickpoints = len(clickdata)
    
    left = np.arange(1, Nclickpoints, 2)
    right = np.arange(2, Nclickpoints, 2)
    
    # vector between left and right VPs
    dx_clickdata = clickdata[left, :] - clickdata[right, :]
    
    # baseline lengths
    l_clickdata = np.hypot(vlr_clickdata[:, 0], vlr_clickdata[:, 1])
    d_clickdata = np.hypot(dx_clickdata[:, 0], dx_clickdata[:, 1])
    
    return l_clickdata, d_clickdata

# <codecell>

def exportdata2csv(implant_recurrens, filename = 'recurrens_paralysis_2013_10_23'):
    
    import csv
    
    varnames = ['stimind', 'F0', 'onsettime_ms', 'ps_onset', 'Q_onset']
    
    alldata = dict()
    
    for casename in implant_recurrens:
        print 'casename: ', casename
        
        # sometimes phonation was determined after stimulation had stopped
        # this sound is due to switching off the stimulation and the flow ramp
        phonation = implant_recurrens[casename]['onsettime_ms'] < 1500

        # sort the un-raveled array
        sortindices = np.argsort( implant_recurrens[casename]['stimind'].ravel() )
        
        alldata[casename] = np.array([np.where(phonation, implant_recurrens[casename][varname], np.nan).ravel()[sortindices] 
                                      for varname in varnames]).T
        
        alldata[casename][:, 0] = implant_recurrens[casename]['stimind'].ravel()[sortindices].astype(int)
        
        csvfilename = '{}.{}.alldata.csv'.format(filename, casename)
    
        with open(csvfilename, 'wt') as f:
            writecsv = csv.writer(f)
            
            # write the header
            writecsv.writerow(varnames)
            
            for row in alldata[casename]:
                writecsv.writerow(row)    

# <codecell>

import time

def datetimestamp(cinefilename, debug = False):

    fname = os.path.basename(cinefilename)

    cleanf = os.path.splitext(fname)[0]
    
    if debug:
        print 'fname: ', fname
        print 'cleanf: ', cleanf

    # all our experiments are on Wednesdays!!!
    # datestring, usec, nsec = 
    items = cleanf.split('Wed')[-1].strip().split('.')
    
    if debug:
        print 'items: ', items
    
    datestring = items[0]

    return time.strptime(datestring, '%b %d %Y %H %M %S')

# <codecell>

def scatterplot(implant_recurrens, casenames, title = 'recurrens'):

    markers = ['o', 's', '*', 'v', '^', '<', '>', 'D', 'p', 'h', '8']
    # markers = ['o'] * 12
    
    allF0 = np.hstack([implant_recurrens[casename]['F0'].ravel() for casename in casenames])
    allA = np.hstack([implant_recurrens[casename]['A_onset'].ravel() for casename in casenames])
    allP = np.hstack([implant_recurrens[casename]['P_onset'].ravel() for casename in casenames])
    
    allps = np.hstack([implant_recurrens[casename]['ps_onset'].ravel() for casename in casenames])
    allQ = np.hstack([implant_recurrens[casename]['Q_onset'].ravel() for casename in casenames])
    
    for casename in casenames:
        plt.close('all')
        plt.figure(figsize = (20, 15))
        
        A = implant_recurrens[casename]['A_onset'].ravel()
        P = implant_recurrens[casename]['P_onset'].ravel()
        
        Q = implant_recurrens[casename]['Q_onset'].ravel()
        ps = implant_recurrens[casename]['ps_onset'].ravel()
        F0 = implant_recurrens[casename]['F0'].ravel()
        
        # plt.plot(implant_recurrens[casename]['ps_onset'].ravel(), 
        #             implant_recurrens[casename]['Q_onset'].ravel(), 'o', 
        #          label = casename)
        
        # plt.plot(implant_recurrens[casename]['F0'].ravel(), A.ravel(), 'o', label = casename, mec = 'None')
        
        plt.scatter(# A, P, 
                    Q, ps,
                    s = F0 * 5, 
                    c = F0,
                    vmin = np.nanmin(allF0), vmax = np.nanmax(allF0),
                    edgecolors = 'None',
                    marker = markers[casenames.index(casename)],
                    label = casename,
                    alpha = 0.7)
        
        cb = plt.colorbar()
        cb.set_label('onset F0 [Hz]')
        
        plt.title(casename)
        plt.xlabel('onset Q [ml/s]')
        # plt.xlabel('onset Bernoulli area [a. u.]')
        plt.ylabel('onset ps [Pa]')
        # plt.ylabel('onset aerodynamic power [W]')
        

        plt.xlim(xmin = 0.9 * np.nanmin(allQ), xmax = 1.1 * np.nanmax(allQ))
        plt.ylim(ymin = 0, ymax = np.nanmax(allps))
        
        # plt.gray()
        
        figname = '{}.ps-Q-F0.{}.pdf'.format(title.replace(' ', '_'), casename)
        # figname = '{}.power-area-F0.{}.pdf'.format(title, casename)
        plt.savefig(figname, orientation = 'landscape', bbox_inches = 'tight')
        

# <codecell>


