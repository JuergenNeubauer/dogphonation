# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import re

# <markdowncell>

# data sets from 5/23/12
# 
# 1.  no SLN/no RLN (this is complete, including phase analysis)
# 2.  no SLN/max RLN  (all complete except phase)
# 3.  max SLN/max RLN (all complete except phase)

# <codecell>

xlsdir = '/extra/InVivoDog/Dinesh/TA_asymmetry/'

# <codecell>

ls -alot $xlsdir/*.xls*

# <codecell>

book = xlrd.open_workbook(filename = os.path.join(xlsdir, 'DONE 5.23.12.corrected.xlsx'))

# <codecell>

book.sheet_names()

# <codecell>

casenames = ['noSLN noRLN',
             'noSLN maxRLN',
             'maxSLN maxRLN']

# <codecell>

for casename in casenames:
    sheet = book.sheet_by_name(casename)
    
    print sheet.name, sheet.nrows, sheet.ncols
    
    for k in range(5):
        print k, sheet.row_values(k)
        
    print

# <codecell>

# Nstimulations = (8 + 1) * (8 + 1)

Nstimulations = [9*9, 10*10, 9*9]

if type(Nstimulations) is not list:
    Nstimulations = [Nstimulations] * len(casenames)
    
print Nstimulations

# <codecell>

TAasymmetry = {casename: dict(Nstimulation = Nstimulation) for (casename, Nstimulation) in zip(casenames, Nstimulations)}

# <codecell>

# misspelling of 'Vibatory' is intentional

varnames = ['Onset Time (sample)', '#Peaks', 'Elapsed Time (samples)', 'Phase Lead', 'Mucosal Amplitude', 'Vibatory Amplitude']
datavar = {varname: {} for varname in varnames}
datavar

# <codecell>

symvars = ['Phase Lead', 'Mucosal Amplitude', 'Vibatory Amplitude']

# <codecell>

for casename in casenames:
    sheet = book.sheet_by_name(casename)

    onset_list = []
    T_list = []
    symmetry_dict = {varname: [] for varname in symvars}
    
    Nstimulation = TAasymmetry[casename]['Nstimulation']
    
    header = sheet.row_values(1)
    varindices = []
    
    for varname in varnames:
        # returns FIRST occurrence index
        try:
            ind = header.index(varname)
        except:
            print 'header: ', header
            print "varname not found in header: ", varname
            ind = None
            
        datavar[varname].update(varind = ind)
    
    for rownum in range(Nstimulation):
        # onsettime_samples, _, Npeaks, Tsamples = sheet.row_values(rownum + 2, start_colx = 1, end_colx = 5)
        row_values = sheet.row_values(rownum + 2)
        
        for varname in varnames:
            datavar[varname].update(val = row_values[datavar[varname]['varind']])
            
        Npeaks = datavar['#Peaks']['val']
        if Npeaks != 4:
            print casename
            print "Npeaks: ", Npeaks
            raise ValueError("Npeaks should be 4, instead: {}".format(Npeaks))
        
        onsettime_samples = datavar['Onset Time (sample)']['val']
        try:
            if onsettime_samples.strip().lower() in ['', 'np']:
                onset_list.append(np.nan)
            else:
                print "onsettime_samples value unexpected: {}".format(onsettime_samples)
                print "stopping: casename: {}, rownum: {}".format(casename, rownum)
                print
                break
        except:
            onset_list.append(int(onsettime_samples))
            
        Tsamples = datavar['Elapsed Time (samples)']['val']
        try:
            if Tsamples.strip().lower() in ['', 'np']:
                T_list.append(np.nan)
            else:
                print "Tsamples value unexpected: {}".format(Tsamples)
                print "stopping: casename: {}, rownum: {}".format(casename, rownum)
                print
                break
        except:
            T_list.append(int(Tsamples))
            
        for varname in symvars:
            val = datavar[varname]['val']
            
            try:
                val = str(val).strip().lower()
            except:
                print casename, varname
                print "problem: not a string: ", val
                break
                
            datavar[varname]['alt val'] = None    
            
            if val in ['l', 'r', 's']:
                datavar[varname]['val'] = val
            elif val in ['']:
                datavar[varname]['val'] = None
            elif 'slight' in val or 'or' in val:
                print "symmetry: '{}'".format(val)
                
                res = re.search("(.*) \((.*)\)", val)
                if res:
                    print 're: ', res.groups()
                
                    datavar[varname]['val'] = res.groups()[0]
                    datavar[varname]['alt val'] = res.groups()[1]
                else:
                    datavar[varname]['val'] = val
                    
                print datavar[varname]
                print
            else:
                print "unknown symmetry: '{}'".format(val)
                print
                datavar[varname]['val'] = None
                
            symmetry_dict[varname].append(datavar[varname].copy())
            
    onsettime_ms = np.array(onset_list) / 50.
    
    F0 = 50. * 4. / np.array(T_list) * 1000.
    
    TAasymmetry[casename].update(F0 = F0, onsettime_ms = onsettime_ms, symmetry = symmetry_dict.copy())

# <codecell>

TAasymmetry

# <codecell>

basedir = "/extra/InVivoDog/InVivoDog_2012_05_23"

expname = "asymmetricTA"

TAasymmetry['noSLN noRLN'].update(hdf5datadir = expname + "/NoSLN NoRLN/data")
TAasymmetry['noSLN maxRLN'].update(hdf5datadir = expname + "/NoSLN MaxRLN/data")
TAasymmetry['maxSLN maxRLN'].update(hdf5datadir = expname + "/MaxSLN MaxRLN/data")

# <codecell>

for casename in casenames:
    hdf5path = TAasymmetry[casename]['hdf5datadir']
    
    fullpath = os.path.join(basedir, hdf5path)
    print fullpath
    print 'path exists? ', os.path.isdir(fullpath)
    
    print glob.glob(fullpath + '/*.hdf5')
    
    print

# <codecell>

min_F0 = np.min([np.nanmin(TAasymmetry[casename]['F0']) for casename in TAasymmetry])
max_F0 = np.max([np.nanmax(TAasymmetry[casename]['F0']) for casename in TAasymmetry])

print "min_F0: {} Hz, max_F0: {} Hz".format(min_F0, max_F0)

# <codecell>

hdf5filename = sorted(glob.glob(fullpath + '/*.hdf5'))

if len(hdf5filename) > 1:
    print "more than one hdf5 data file found, take the latest one"
    print hdf5filename
    
hdf5filename = hdf5filename[-1]
print "data file: ", hdf5filename

d = dogdata.DogData(datadir = fullpath, datafile = os.path.basename(hdf5filename))

d.get_all_data()

# <codecell>

plt.close('all')

meanQ = (np.mean(d.allQ, axis = 0))
stdQ = (np.std(d.allQ, axis = 0))

plt.plot(meanQ)

plt.plot(meanQ - stdQ, 'r.-')
plt.plot(meanQ + stdQ, 'r.-')

plt.show()

# <codecell>

print casename
d.a_rellevels

# <codecell>

%run -i 'tools for implant analysis.py'

# <codecell>

# no correction needed for this date, no iso-amp used at this time, see also raw flow rate data: max is around 1600 ml/s
    
Voffset_ps = 0.0 #
isoampgain_ps = 1.0 #
    
Voffset_Q = 0.0 #
isoampgain_Q = 1.0

# <codecell>

# left right TA asymmetry
#
# relative levels from column 4 for left TA
# and column 5 for right TA

num_leftTA = 4
num_rightTA = 5

relative_nerve_levels = [('leftTA', num_leftTA), 
                         ('rightTA', num_rightTA)]

min_ps, min_Q = getonsetdata(basedir, TAasymmetry, relative_nerve_levels)

# <codecell>

print TAasymmetry[casename].keys()
print symvars

# <codecell>

def replace(s):
    if s == None:
        return 'no'
    else:
        return s

savevarnames = ['phase lead', 'mucosal amp', 'vibratory amp']
    
for casename in casenames:
    for varname, savevarname in zip(symvars, savevarnames):
        
        stimind = TAasymmetry[casename]['stimind'].astype(np.int)
        
        l_val = [item['val'] for item in TAasymmetry[casename]['symmetry'][varname]]
        l_altval = [item['alt val'] for item in TAasymmetry[casename]['symmetry'][varname]]
        
        l_val = map(replace, l_val)
        l_altval = map(replace, l_altval)
        
        TAasymmetry[casename].update({savevarname: np.array(l_val)[stimind]})
        savevarname += ' alt'
        TAasymmetry[casename].update({savevarname: np.array(l_altval)[stimind]})
        
    del TAasymmetry[casename]['symmetry']

# <codecell>

TAasymmetry

# <codecell>

sidecodes = [('l', -1), ('slight l', -0.5), ('or l', -2),
             ('s', 0), ('or s', 10),
             ('slight r', 0.5), ('r', 1), ('or r', 2)]

# <codecell>

for casename in casenames:
    print casename
    
    for varname in savevarnames:
        print varname

        var = TAasymmetry[casename][varname]
        varalt = TAasymmetry[casename][varname + ' alt']
        
        numsym = np.ones_like(var, dtype = np.float) * np.nan
        
        for charside, numside in sidecodes:
            numsym = np.where(var == charside, numside, numsym)
            numsym = np.where(varalt == charside, numside, numsym)
        
        TAasymmetry[casename]['onset ' + varname] = numsym
        
        print numsym
        print numsym.shape
        print TAasymmetry[casename]['Nstimulation']

# <codecell>

Bernoulli_Power(TAasymmetry)

# <codecell>

for casename in casenames:
    phonation = TAasymmetry[casename]['onsettime_ms'] < 1500
    
    if not np.all(phonation):
        print casename
        print "errors in onset time"

# <codecell>

plotsymmetry(TAasymmetry, symvars = savevarnames, name_paralysis = 'TAasymmetry')

# <codecell>

plotonsetdata(TAasymmetry, name_paralysis = 'TAasymmetry', 
              F0_normalized = True, 
              ps_normalized = True, 
              Q_normalized = True)

# <codecell>

%run -i "tools for implant analysis.py"

# <codecell>

scatterplot(TAasymmetry, casenames, title = 'TAasymmetry')

# <codecell>

allF0 = np.hstack([TAasymmetry[casename]['F0'].ravel() for casename in casenames])
allA = np.hstack([TAasymmetry[casename]['A_onset'].ravel() for casename in casenames])
allP = np.hstack([TAasymmetry[casename]['P_onset'].ravel() for casename in casenames])

allps = np.hstack([TAasymmetry[casename]['ps_onset'].ravel() for casename in casenames])
allQ = np.hstack([TAasymmetry[casename]['Q_onset'].ravel() for casename in casenames])

# <codecell>

plt.close('all')

plt.scatter(allA**(5/2.) * allF0**3, allP, s = allF0, c = allF0)

plt.xlabel('A')
plt.ylabel('P')

plt.show()

# <codecell>

# landmarks at rest and onset clicked by Eli
# if onset didn't occur, used the last frame in recording
# see strainanalysis.py
landmarkdir = "/extra/InVivoDog/Elazar/results/05_23_2012_asymmetricTA/"

print "%s exists? " % landmarkdir, os.path.isdir(landmarkdir)

# <codecell>

!ls -alot $landmarkdir

# <codecell>

casenames

# <codecell>

video_casenames = ['NoSLN NoRLN', 'NoSLN MaxRLN', 'MaxSLN MaxRLN']

# <codecell>

clickfiles = dict()

for casename, videocasename in zip(casenames, video_casenames):
    clickfiles[casename] = sorted(glob.glob(os.path.join(landmarkdir, '*{}*.npz'.format(videocasename))))

# <codecell>

clickfiles

# <codecell>

def distances(clickdata):
    # baseline vectors between anterior landmark and VP on left and right sides
    vlr_clickdata = clickdata[1:] - clickdata[0]
    # vector between left and right VPs
    dx_clickdata = clickdata[1] - clickdata[2]
    
    # baseline lengths
    l_clickdata = np.hypot(vlr_clickdata[:, 0], vlr_clickdata[:, 1])
    d_clickdata = np.hypot(dx_clickdata[0], dx_clickdata[1])
    
    return l_clickdata, d_clickdata

# <codecell>

strainvarnames = ['lstrain', 'rstrain', 'dVP']

for casename in casenames:
    print casename
    
    strains = []
    stimlevelindex = []
    
    for clickfile in clickfiles[casename]:
        data = np.load(clickfile)
        
        baseline = data['baseline_pos']
        onset = data['onset_pos']
        stimlevelindex.append(data['stimlevelindex'].tolist())

        l_baseline, d_baseline = distances(baseline)
        l_onset, d_onset = distances(onset)
        
        leftstrain, rightstrain = (l_onset - l_baseline) / l_baseline * 100.0
        # dVPrel = (d_onset - d_baseline) / d_baseline * 100.0
        dVPrel = d_onset / d_baseline * 100.0
        
        if np.any(np.isnan([leftstrain, rightstrain, dVPrel])):
            print 'clickfile: ', clickfile
            for item in data.files:
                print "{}: {}".format(item, data[item])
            print
                
        strains.append([leftstrain, rightstrain, dVPrel])
            
    leftstrain = [item[0] for item in strains]
    rightstrain = [item[1] for item in strains]
    dVPrel = [item[2] for item in strains]
            
    stimind = TAasymmetry[casename]['stimind'].astype(np.int)
    
    strainlists = [leftstrain, rightstrain, dVPrel]
    
    for savevarname, strainlist in zip(strainvarnames, strainlists):
        TAasymmetry[casename].update({savevarname: np.array(strainlist)[stimind]})

# <codecell>

TAasymmetry

# <codecell>

def plotstrains(TAasymmetry, name_paralysis = 'TAasymmetry', normalized = True):
    
    min_lstrain = np.nanmin([np.nanmin(TAasymmetry[casename]['lstrain']) for casename in TAasymmetry])
    max_lstrain = np.nanmax([np.nanmax(TAasymmetry[casename]['lstrain']) for casename in TAasymmetry])
    
    print "min_lstrain: {}, max_lstrain: {}".format(min_lstrain, max_lstrain)
    
    min_rstrain = np.nanmin([np.nanmin(TAasymmetry[casename]['rstrain']) for casename in TAasymmetry])
    max_rstrain = np.nanmax([np.nanmax(TAasymmetry[casename]['rstrain']) for casename in TAasymmetry])
    
    print "min_rstrain: {}, max_rstrain: {}".format(min_rstrain, max_rstrain)
    
    min_dVP = np.nanmin([np.nanmin(TAasymmetry[casename]['dVP']) for casename in TAasymmetry])
    max_dVP = np.nanmax([np.nanmax(TAasymmetry[casename]['dVP']) for casename in TAasymmetry])
    
    print "min_dVP: {}, max_dVP: {}".format(min_dVP, max_dVP)
    
    plt.close('all')
    
    for casename in TAasymmetry:
        xlabel = TAasymmetry[casename]['nerve_xaxis']
        ylabel = TAasymmetry[casename]['nerve_yaxis']
        
        lstrain = TAasymmetry[casename]['lstrain']
        rstrain = TAasymmetry[casename]['rstrain']
        dVP = TAasymmetry[casename]['dVP']
        
        ################################################################################################
        
        try:
            plt.clf()
        except:
            pass
        
        plt.imshow(lstrain)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        textnorm = ''
        if normalized:
            plt.clim(vmin = min_lstrain, vmax = max_lstrain)
            textnorm = '.Normalized'
            
        plt.title("{}: {}".format(name_paralysis, casename))
        
        cb = plt.colorbar()
        cb.set_label('left strain [%]')
        
        plt.grid(False)
    
        savename = "{}.lstrain{}.{}.pdf".format(name_paralysis.replace(' ', '_'), textnorm, casename)
        
        plt.savefig(savename, orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)
        
        ################################################################################################
        
        plt.clf()
        
        plt.imshow(rstrain)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        textnorm = ''
        if normalized:
            plt.clim(vmin = min_rstrain, vmax = max_rstrain)
            textnorm = '.Normalized'
            
        plt.title("{}: {}".format(name_paralysis, casename))
        
        cb = plt.colorbar()
        cb.set_label('right strain [%]')
        
        plt.grid(False)
    
        savename = "{}.rstrain{}.{}.pdf".format(name_paralysis.replace(' ', '_'), textnorm, casename)
        
        plt.savefig(savename, orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)
    
        ################################################################################################
        
        plt.clf()
        
        plt.imshow(dVP)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        textnorm = ''
        if normalized:
            plt.clim(vmin = min_dVP, vmax = max_dVP)
            textnorm = '.Normalized'
            
        plt.title("{}: {}".format(name_paralysis, casename))
        
        cb = plt.colorbar()
        cb.set_label('Dvp [%]')
        
        plt.grid(False)
    
        savename = "{}.dVP{}.{}.pdf".format(name_paralysis.replace(' ', '_'), textnorm, casename)
        
        plt.savefig(savename, orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

plotstrains(TAasymmetry, name_paralysis = 'TAasymmetry', normalized = False)

# <codecell>

TAasymmetry['noSLN noRLN']['lstrain']

# <codecell>

allF0 = np.hstack([TAasymmetry[casename]['F0'].ravel() for casename in casenames])
allA = np.hstack([TAasymmetry[casename]['A_onset'].ravel() for casename in casenames])
allP = np.hstack([TAasymmetry[casename]['P_onset'].ravel() for casename in casenames])

allps = np.hstack([TAasymmetry[casename]['ps_onset'].ravel() for casename in casenames])
allQ = np.hstack([TAasymmetry[casename]['Q_onset'].ravel() for casename in casenames])

alllstrain = np.hstack([TAasymmetry[casename]['lstrain'].ravel() for casename in casenames])
allrstrain = np.hstack([TAasymmetry[casename]['rstrain'].ravel() for casename in casenames])
alldVP = np.hstack([TAasymmetry[casename]['dVP'].ravel() for casename in casenames])

# <codecell>

plt.close('all')

scatter = plt.scatter(alllstrain, allF0, s = 100, c = alldVP, alpha = 0.7)

plt.xlabel('left strain [%]')
plt.ylabel('F0 [Hz]')

cb = plt.colorbar(mappable = scatter)
cb.set_label('Dvp [%]')

plt.xlim(-10, 25)
plt.ylim(0, 500)

plt.savefig('TAasymmetry.F0-strain_Dvp.pdf', orientation = 'landscape', bbox_inches = 'tight')

plt.show()

# <codecell>

plt.close('all')

plt.scatter(allA**(5/2.) * allF0**3, allP, s = allF0, c = allF0)

plt.xlabel('A')
plt.ylabel('P')

plt.show()

