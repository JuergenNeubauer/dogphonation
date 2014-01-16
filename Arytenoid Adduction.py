# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os, csv, glob
import numpy as np

# %pylab inline

import matplotlib as mpl
# use the backend for inline plots in the IPython Notebook
# so then don't need to use magic pylab with inline option
mpl.use('module://IPython.zmq.pylab.backend_inline')

import matplotlib.pyplot as plt

print "Matplotlib will use backend: ", mpl.get_backend()
print "Pyplot will use backend: ", plt.get_backend()

sys.path.append("/extra/InVivoDog/python/cine/tools")

# DogData imports matplotlib and tries to set the backend to GTKAgg
# need to change this behavior!!!
from dogdata import DogData

# <markdowncell>

# # Eli's landmark finding results: landmarks and strain_onset images and data

# <codecell>

ls -alt /extra/InVivoDog/Dinesh/Arytenoid_Adduction

# <markdowncell>

# # analog data: acoustics, pressure, flow

# <codecell>

ls -alt /extra/InVivoDog/InVivoDog_2012_02_22/

# <codecell>

ls -alt /extra/InVivoDog/InVivoDog_2012_04_04/

# <codecell>

datadir = "/extra/InVivoDog/InVivoDog_2012_02_22/adduction_suture_left_VP_LeftRLN_paralysis/"

# <codecell>

ls -alt $datadir

# <codecell>

r = !ls $datadir

r

# <codecell>

{'Exp%02d' % (ni+1): n for ni, n in enumerate(r)}

# <codecell>

datafiles = {"Exp01": "adduction_exp01_SLN-RLN.hdf5",
             "Exp03": "adduction_exp03_SLN-RLN.hdf5",
             "Exp05": "adduction_exp05_SLN-RLN.hdf5",
             "Exp07": "adduction_exp07_SLN-RLN.hdf5",
             "Exp09": "adduction_exp09_SLN-RLN.hdf5",
             "Exp02": "adduction_exp02_rightSLN-RLN.hdf5",
             "Exp04": "adduction_exp04_rightSLN-RLN.hdf5",
             "Exp06": "adduction_exp06_rightSLN-RLN.hdf5",
             "Exp08": "adduction_exp08_rightSLN-RLN.hdf5",
             "Exp10": "adduction_exp10_rightSLN-RLN.hdf5"}

adduction_conditions = sorted(datafiles.keys())

# <codecell>

adduction_conditions_names = ['adduction01_SLN-RLN',
                              'adduction01_rightSLN-RLN',
                              'adduction02_SLN-RLN',
                              'adduction02_rightSLN-RLN',
                              'adduction03_SLN-RLN',
                              'adduction03_rightSLN-RLN',
                              'adduction04_SLN-RLN',
                              'adduction04_rightSLN-RLN',
                              'adduction05_SLN-RLN',
                              'adduction05_rightSLN-RLN']

# <markdowncell>

# # onset time and frequency

# <codecell>

ls /extra/InVivoDog/Dinesh/Arytenoid_Adduction/*.xlsx

# <codecell>

import xlrd

excel_dir = "/extra/InVivoDog/Dinesh/Arytenoid_Adduction/"

excel_onsetfile = 'Adduction_LeftVP_LeftRLN_paralysis 2_22_12 FINAL.xlsx'

# <codecell>

excel_filename = os.path.join(excel_dir, excel_onsetfile)
print "opening Excel file: ", excel_filename
print
book = xlrd.open_workbook(filename = excel_filename)

print 'sheets: ', book.sheet_names()
print
for sheet in book.sheets():
    print "# %d: %s: #rows: %d #cols: %d" % (sheet.number, sheet.name, sheet.nrows, sheet.ncols)

# <codecell>

F0 = {} # onset frequency in Hz
onset_time = {} # onset time in milliseconds

# <codecell>

for adductnum, (adductcond, adductcondname) in enumerate(zip(adduction_conditions, adduction_conditions_names)):
    dogdata = DogData(datadir = datadir, datafile = datafiles[adductcond])
    print "working on: ", dogdata.datafilename
    
    Nrecnums = dogdata.Nrecnums
    print "Nrecnums: ", Nrecnums
    
    sheet_adductcond = book.sheet_by_name(adductcond)
    print "getting data from Excel sheet: ", sheet_adductcond.name
    print
    
    onset_time[ adductcond ] = []
    F0[ adductcond ] = []
    
    for recnum in range(Nrecnums):
        # File #, Onset Time (sample), Onset time (ms), #Peaks, End Time (samples), Elapsed Time (samples), Elapsed Time (ms), F0
        sheet_adductcond.row_values(3 + recnum, start_colx = 1, end_colx =)

# <codecell>


# <codecell>

F0 = np.array(F0)
onset = np.array(onset)

ps_onset = np.ones(onset.shape) * np.nan
Q_onset = np.ones(onset.shape) * np.nan

ps_noonset = np.ones(onset.shape) * np.nan
ps_noonset_maxind = np.zeros(onset.shape)

Q_noonset = np.ones(onset.shape) * np.nan

# Bernoulli equivalent area A proportional to Q / sqrt(p)
A_onset = np.ones(onset.shape) * np.nan
A_noonset = np.ones(onset.shape) * np.nan

# <codecell>

for TAnum, TAcond in enumerate(TAconditions):
    dogdata = DogData(datadir = datadir, datafile = datafiles[TAcond])
    print "working on: ", dogdata.datafilename
    dogdata.get_all_data()
    
    time = dogdata.time_psQ * 1000.0 # onset time is given in milliseconds
    
    for stimind, datarow in enumerate(dogdata.allps):
        if not np.isnan(onset[stimind, TAnum]):
            ps_onset[stimind, TAnum] = np.interp(onset[stimind, TAnum], time, datarow)
        else:
            ps_noonset[stimind, TAnum] = datarow.max()
            ## ps_noonset_maxind does not make sense because of slight delay between flow ramp and pressure response
            ## therefore while flow is already decreasing after the end of the ramp, the pressure still rises for a short time
            # ps_noonset_maxind[stimind, TAnum] = datarow.argmax()

    for stimind, datarow in enumerate(dogdata.allQ):
        if not np.isnan(onset[stimind, TAnum]):
            Q_onset[stimind, TAnum] = np.interp(onset[stimind, TAnum], time, datarow)
        else:
            # the maximum flow rate should always be constant as set by the flow controller for the max of the flow ramp
            Q_noonset[stimind, TAnum] = datarow.max()
            ## not useful, see comment on ps_noonset_maxind above
            # Q_noonset[stimind, TAnum] = datarow[ps_noonset_maxind[stimind, TAnum]]

A_onset = Q_onset / np.sqrt(ps_onset)
A_noonset = Q_noonset / np.sqrt(ps_noonset)

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

# landmarks at rest and onset clicked by Eli
# if onset didn't occur, used the last frame in recording
# see strainanalysis.py
landmarkdir = "/extra/InVivoDog/Dinesh/Arytenoid_Adduction/02_22_2012_Adduction/"
# landmarkdir = "/extra/InVivoDog/python/cine/results_save"

# <codecell>

clickfiles = sorted(glob.glob(os.path.join(landmarkdir, '*.npz')))

# <codecell>

sorted([os.path.basename(item) for item in clickfiles])

# <codecell>

strains = {}

for TAnum, TAcond in enumerate(TAconditions):
    strains[TAcond] = []

    TAcond_name = datafiles[TAcond].split()[0]

    TAcond_landmarkfiles = sorted(
        [item for item in clickfiles if os.path.basename(item).startswith(TAcond_name)])

    for clickfile in TAcond_landmarkfiles:
        data = np.load(clickfile)

        if data['TAconditionindex'] != TAnum:
            print "Error: wrong TA condition index"
            continue

        baseline = data['baseline_pos']
        if len(baseline) < 3:
            print "ERROR: baseline: #clicks < 3: ", clickfile
            strains.append([np.nan, np.nan, np.nan])
            continue
        onset = data['onset_pos']
        if len(onset) < 3:
            print "ERROR: onset: #clicks < 3: ", clickfile
            strains.append([np.nan, np.nan, np.nan])
            continue

        l_baseline, d_baseline = distances(baseline)
        l_onset, d_onset = distances(onset)

        leftstrain, rightstrain = (l_onset - l_baseline) / l_baseline * 100.0
        # dVPrel = (d_onset - d_baseline) / d_baseline * 100.0
        dVPrel = d_onset / d_baseline * 100.0

        # FIX outlier by hand: TA condition 2, clickfile 11 (counting starts from 0)
        if TAnum == 2:
            if clickfile == TAcond_landmarkfiles[11]:
                leftstrain = strains[TAcond][9][0] * 1.05
                rightstrain = strains[TAcond][9][1] * 1.07
                dVPrel = strains[TAcond][9][2] * 0.94

        strains[TAcond].append([leftstrain, rightstrain, dVPrel])

# <codecell>

F0_plot = {}
ps_plot = {}
Q_plot = {}
A_plot = {}

ps_np_plot = {}
Q_np_plot = {}
A_np_plot = {}

lstrain_plot = {}
rstrain_plot = {}
dVP_plot = {}

lstrain_np_plot = {}
rstrain_np_plot = {}
dVP_np_plot = {}

Nlevels = dogdata.Nlevels
# dogdata.nervenamelist
# ['left SLN', 'right SLN', 'left RLN', 'right RLN', 'left TA', 'right TA']
# we need SLN versus RLN trunk
rellevels = dogdata.a_rellevels[:, [0, 2]]

for TAnum, TAcond in enumerate(TAconditions):
    F0_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    ps_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    Q_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    A_plot[TAcond] = np.zeros((Nlevels, Nlevels))

    ps_np_plot[TAcond] = np.ones((Nlevels, Nlevels)) * np.nan
    Q_np_plot[TAcond] = np.ones((Nlevels, Nlevels)) * np.nan
    A_np_plot[TAcond] = np.ones((Nlevels, Nlevels)) * np.nan

    lstrain_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    rstrain_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    dVP_plot[TAcond] = np.zeros((Nlevels, Nlevels))

    lstrain_np_plot[TAcond] = np.ones((Nlevels, Nlevels)) * np.nan
    rstrain_np_plot[TAcond] = np.ones((Nlevels, Nlevels)) * np.nan
    dVP_np_plot[TAcond] = np.ones((Nlevels, Nlevels)) * np.nan

    for stimind, (SLNlevel, RLNlevel) in enumerate(rellevels):
        # SLN: row index, i.e. y-axis in plot
        # RLN: column index, i.e. x-axis in plot
        F0_plot[TAcond][SLNlevel, RLNlevel] = F0[stimind, TAnum]
        ps_plot[TAcond][SLNlevel, RLNlevel] = ps_onset[stimind, TAnum]
        Q_plot[TAcond][SLNlevel, RLNlevel] = Q_onset[stimind, TAnum]
        A_plot[TAcond][SLNlevel, RLNlevel] = A_onset[stimind, TAnum]

        lstrain_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][0]
        rstrain_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][1]
        dVP_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][2]

        ps_np_plot[TAcond][SLNlevel, RLNlevel] = ps_noonset[stimind, TAnum]
        Q_np_plot[TAcond][SLNlevel, RLNlevel] = Q_noonset[stimind, TAnum]
        A_np_plot[TAcond][SLNlevel, RLNlevel] = A_noonset[stimind, TAnum]
        
        if np.isnan(F0_plot[TAcond][SLNlevel, RLNlevel]):
            lstrain_plot[TAcond][SLNlevel, RLNlevel] = np.nan
            rstrain_plot[TAcond][SLNlevel, RLNlevel] = np.nan
            dVP_plot[TAcond][SLNlevel, RLNlevel] = np.nan
            
            lstrain_np_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][0]
            rstrain_np_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][1]
            dVP_np_plot[TAcond][SLNlevel, RLNlevel] = strains[TAcond][stimind][2]           

# <codecell>

all_F0 = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_ps = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_Q = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_A = np.zeros( (Nlevels, len(TAconditions), Nlevels) )

all_ps_np = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_Q_np = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_A_np = np.zeros( (Nlevels, len(TAconditions), Nlevels) )

all_lstrain = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_rstrain = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_dVP = np.zeros( (Nlevels, len(TAconditions), Nlevels) )

all_lstrain_np = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_rstrain_np = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_dVP_np = np.zeros( (Nlevels, len(TAconditions), Nlevels) )

for TAnum, TAcond in enumerate(TAconditions):
    all_F0[:, TAnum, :] = F0_plot[TAcond] # SLNlevel, TAlevel, trunkRLNlevel
    all_ps[:, TAnum, :] = ps_plot[TAcond]
    all_Q[:, TAnum, :] = Q_plot[TAcond]
    all_A[:, TAnum, :] = A_plot[TAcond]

    all_ps_np[:, TAnum, :] = ps_np_plot[TAcond]
    all_Q_np[:, TAnum, :] = Q_np_plot[TAcond]
    all_A_np[:, TAnum, :] = A_np_plot[TAcond]

    all_lstrain[:, TAnum, :] = lstrain_plot[TAcond]
    all_rstrain[:, TAnum, :] = rstrain_plot[TAcond]
    all_dVP[:, TAnum, :] = dVP_plot[TAcond]
    
    all_lstrain_np[:, TAnum, :] = lstrain_np_plot[TAcond]
    all_rstrain_np[:, TAnum, :] = rstrain_np_plot[TAcond]
    all_dVP_np[:, TAnum, :] = dVP_np_plot[TAcond]    

# <codecell>

from IPython.display import HTML
def webframe(urlstring, width = 1000, height = 500):
        return HTML("<iframe src=%s width=%d height=%d></iframe>" % (urlstring, width, height))

# <codecell>

webframe("http://asadl.org/jasa/resource/1/jasman/v129/i4/p2253_s1")

# <codecell>

webframe("http://scitation.aip.org/getpdf/servlet/GetPDFServlet?filetype=pdf&id=JASMAN000129000004002253000001&idtype=cvips&doi=10.1121/1.3552874&prog=normal")

# <markdowncell>

# Comparison with Herbst (JASA)
# =============================
# 
# prediction/observation from human subjects:
# -------------------------------------------
# 
# * division into cartilagenous and membranous ADduction and ABduction
# * low TA: membraneous ABduction
# * low trunk RLN, i.e. LCA/IA: cartilagenous ABduction
# 
# vocal registers
# ---------------
# 
# * low TA and any trunk RLN: falsetto register: high F0
# * high TA and any trunk RLN: chest register: low F0
# 
# in vivo canine experiments
# --------------------------
# 
# * TA level 0
#     * NO TA
#     * low F0
# 
# 
# * TA level 1
#     * threshold TA
#     * high F0
# 

# <codecell>

mpl.rcParams['figure.figsize'] = (10, 5)

# <codecell>

# all_F0 [ SLNlevel, TAlevel, trunkRLNlevel ]

minF0 = np.nanmin(all_F0)
maxF0 = np.nanmax(all_F0)

SLNlevel = 0
plt.subplot(1,4,1)
plt.imshow(all_F0[SLNlevel, :, :].T, vmin = minF0, vmax = maxF0)
plt.title("SLN: %s" % SLNlevel)

plt.ylabel('LCA/IA level')
plt.xlabel('TA level')

SLNlevel = 1
plt.subplot(1,4,2)
plt.imshow(all_F0[SLNlevel, :, :].T, vmin = minF0, vmax = maxF0)
plt.title("SLN: %s" % SLNlevel)

SLNlevel = 4
plt.subplot(1,4,3)
plt.imshow(all_F0[SLNlevel, :, :].T, vmin = minF0, vmax = maxF0)
plt.title("SLN: %s" % SLNlevel)

SLNlevel = 6
plt.subplot(1,4,4)
plt.imshow(all_F0[SLNlevel, :, :].T, vmin = minF0, vmax = maxF0)
plt.title("SLN: %s" % SLNlevel)

# cb = plt.colorbar()
# cb.set_label('frequency [Hz]')

plt.show()

# <markdowncell>

# Comparison to RLN stimulation
# ==============================
# 
# When we stimulate the whole RLN, there is an unknown stereotypical excitation of the different motor neurons for the TAs, LCA, and IA. In general, I expect that the thicker nerve fibers (faster conduction velocity, larger diameter, larger node of Ranvier, higher impedance, therefore, lower excitation threshold) are excited first and to a larger degree. So the hierarchy would always be TA (TA lateral > TA medial) > LCA > IA. So in the above plots we would follow a route (not necessarily linear route) where the TA level is much larger than the LCA/IA level.
# 
# Nevertheless, at some point for increasing SLN we have to cross over the high-frequency band. This should be observable in SLN-RLN experiments.

# <codecell>

webframe("http://jneubaue.bol.ucla.edu/publications/JAS01401.pdf#page=4", width = 1100)

# <markdowncell>

# Scatter plots
# ==============
# 
# Onset frequency versus strain: 
# ------------------------------
# 
# Suggests at least two clusters of F0 values: a linear increase in a lower 'register' cluster and another linear increase above a critical strain in a higher 'register' cluster

# <codecell>

falsetto = all_F0.ravel() > 385
chest = all_F0.ravel() <= 385

# <codecell>

fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(all_lstrain.ravel()[chest], all_F0.ravel()[chest], 'g.', ms = 20)
plt.plot(all_lstrain.ravel()[falsetto], all_F0.ravel()[falsetto], 'r+', ms = 20, mec = 'red', mfc = 'None')

plt.plot(all_lstrain_np.ravel(), np.ones_like(all_lstrain_np).ravel() * 20, '|', ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

plt.xlim(xmax = 42)

plt.xlabel('strain [%]')
plt.ylabel('F0 [Hz]')
# plt.title('onset frequency')

plt.legend(loc = 'upper left', numpoints = 1)

plt.savefig('F0_strain.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
plt.show()

# <codecell>

# fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(all_dVP.ravel()[chest], all_ps.ravel()[chest], 'g.', ms = 20)
plt.plot(all_dVP.ravel()[falsetto], all_ps.ravel()[falsetto], 'r.', ms = 20)

plt.plot(all_dVP_np.ravel(), np.ones_like(all_dVP_np).ravel() * 20, '|', ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

plt.xlim(xmin = 10, xmax = 110)

plt.xlabel('dVP [%]')
plt.ylabel('ps [Pa]')
# plt.title('onset frequency')

plt.legend(loc = 'upper right', numpoints = 1)

if True:
    plt.savefig('ps_Dvp.pdf', orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)
plt.show()

# <markdowncell>

# derived from hemilarynx model, although should also hold for bilateral vocal folds
# 
# at phonation onset threshold: WHY? this relationship should hold everywhere, no?
# 
# $$ u_j \propto \omega_0 \sqrt{g} $$
# 
# $$ p_s \propto \omega_0^2 g $$
# 
# $$ Q \propto u_j g $$
# 
# $$ A = \frac{Q}{\sqrt{p_s}} \propto g   $$

# <codecell>

from IPython.display import display, Math, Latex
display(Math(r'F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i k} dx'))

# <codecell>

# fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(all_dVP.ravel()[chest] * all_F0.ravel()[chest]**2, all_ps.ravel()[chest], 'g.', ms = 20)
plt.plot(all_dVP.ravel()[falsetto] * all_F0.ravel()[falsetto]**2, all_ps.ravel()[falsetto], 'r.', ms = 20)

# plt.plot(all_dVP_np.ravel(), np.ones_like(all_dVP_np).ravel() * 20, '|', ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

plt.xlim(xmax = 2.0e7)

plt.xlabel('$d_{VP} * F_0^2$')
plt.ylabel('$p_s [Pa]$')
# plt.title('onset frequency')

plt.legend(loc = 'upper right', numpoints = 1)

if False:
    plt.savefig('ps_Dvp.pdf', orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)
plt.show()

# <codecell>

# fig, ax = plt.subplots(figsize = (16, 12))

cmap = mpl.colors.ListedColormap(['c', 'b', 'g', 'm', 'r'])

plt.scatter(all_dVP, all_ps, s = all_F0 * 0.35, marker = 'o', c = all_F0, edgecolors = 'None', cmap = cmap)
cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.plot(all_dVP_np.ravel(), all_ps_np.ravel(), '*', ms = 15, mec = 'black', mfc = 'None', label = 'no onset', zorder = 1)

plt.xlim(xmin = 10, xmax = 110)
plt.ylim(ymin = 0, ymax = 2500)

plt.xlabel('Dvp [%]')
plt.ylabel('ps [Pa]')

plt.legend(loc = 'upper left', bbox_to_anchor = (0.5, 0.9), numpoints = 1)

if True:
    plt.savefig('ps_Dvp_F0.pdf', orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>

# fig, ax = plt.subplots(figsize = (16, 12))

plt.scatter(all_dVP * all_F0**2, all_ps, s = 50, # s = all_F0 * 0.35, 
            marker = 'o', c = all_F0, edgecolors = 'None', cmap = cmap)

cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.xlim(xmin = 0, # xmax = 5.5e6)
         xmax = 2.0e7)
plt.ylim(ymin = 0, ymax = 2200)

plt.xlabel('Dvp * F0^2 [a.u.]')
plt.ylabel('ps [Pa]')

plt.legend(loc = 'upper left', bbox_to_anchor = (0.5, 0.9), numpoints = 1)

if False:
    plt.savefig('ps_Dvp_F0square.pdf', orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>

# fig, ax = plt.subplots(figsize = (16, 12))

plt.scatter(all_F0, all_lstrain, s = 50, # s = all_F0 * 0.35, 
            marker = 'o', c = all_F0, edgecolors = 'None', cmap = cmap)

cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.xlim(xmin = 50, xmax = 800)
plt.ylim(ymin = -15, ymax = 45)

plt.xlabel('F0 [Hz]')
plt.ylabel('left strain [%]')

plt.legend(loc = 'upper left', bbox_to_anchor = (0.5, 0.9), numpoints = 1)

if False:
    plt.savefig('ps_Dvp_F0square.pdf', orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>

webframe("http://jneubaue.bol.ucla.edu/publications/JAS02279.pdf#page=6", width = 1300)

# <codecell>

fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(all_ps.ravel()[chest], all_F0.ravel()[chest], 'g.', ms = 20)
plt.plot(all_ps.ravel()[falsetto], all_F0.ravel()[falsetto], 'r+', ms = 20, mec = 'red', mfc = 'None')

plt.plot(all_ps_np.ravel(), np.ones_like(all_ps_np).ravel() * 20, '|', ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

plt.xlabel('ps [Pa]')
plt.ylabel('F0 [Hz]')

plt.legend(loc = 'upper left', numpoints = 1)

plt.savefig('F0_ps.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(all_ps[:, 0, :].ravel(), all_F0[:, 0, :].ravel(), 'c*', ms = 10, mec = 'k', mfc = 'None', label = 'TA 0')
plt.plot(all_ps[:, 1, :].ravel(), all_F0[:, 1, :].ravel(), 'b+', ms = 10, mec = 'b', mfc = 'None', label = 'TA 1')
plt.plot(all_ps[:, 2, :].ravel(), all_F0[:, 2, :].ravel(), 'g^', ms = 10, mec = 'g', mfc = 'None', label = 'TA 2')
plt.plot(all_ps[:, 3, :].ravel(), all_F0[:, 3, :].ravel(), 'mv', ms = 10, mec = 'm', mfc = 'None', label = 'TA 3')
plt.plot(all_ps[:, 4, :].ravel(), all_F0[:, 4, :].ravel(), 'ro', ms = 10, mec = 'r', mfc = 'None', label = 'TA 4')

plt.plot(all_ps_np.ravel(), np.ones_like(all_ps_np).ravel() * 50, '|', ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

plt.xlabel('ps [Pa]')
plt.ylabel('F0 [Hz]')

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), numpoints = 1)

plt.savefig('F0_ps_TAcolored.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>

# all_F0 [ SLNlevel, TAlevel, trunkRLNlevel ]

plt.plot(all_lstrain[:, 0, :].ravel(), all_F0[:, 0, :].ravel(), 'c*', ms = 10, mec = 'c', mfc = 'None', label = 'TA 0')
plt.plot(all_lstrain[:, 1, :].ravel(), all_F0[:, 1, :].ravel(), 'b+', ms = 10, mec = 'b', mfc = 'None', label = 'TA 1')
plt.plot(all_lstrain[:, 2, :].ravel(), all_F0[:, 2, :].ravel(), 'g^', ms = 10, mec = 'g', mfc = 'None', label = 'TA 2')
plt.plot(all_lstrain[:, 3, :].ravel(), all_F0[:, 3, :].ravel(), 'mv', ms = 10, mec = 'm', mfc = 'None', label = 'TA 3')
plt.plot(all_lstrain[:, 4, :].ravel(), all_F0[:, 4, :].ravel(), 'ro', ms = 10, mec = 'r', mfc = 'None', label = 'TA 4')

plt.plot(all_lstrain_np.ravel(), np.ones_like(all_lstrain_np).ravel() * 50, '|', mec = 'k', mfc = 'None', label = 'no onset')

plt.xlim(xmax = 42)

plt.xlabel('strain [%]')
plt.ylabel('F0 [Hz]')
# plt.title('onset frequency')

plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), numpoints = 1)

plt.show()

# <markdowncell>

# Onset frequency versus vocal process distance:
# ---------------------------------------------
# 
# NO separation possible

# <codecell>

fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(all_dVP.ravel()[chest], all_F0.ravel()[chest], 'g.', ms = 20)
plt.plot(all_dVP.ravel()[falsetto], all_F0.ravel()[falsetto], 'r+', ms = 20, mec = 'red', mfc = 'None')

plt.plot(all_dVP_np.ravel(), np.ones_like(all_dVP_np).ravel() * 50, '|', ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

plt.xlabel('Dvp [%]')
plt.ylabel('F0 [Hz]')
# plt.title('onset frequency')

plt.legend(loc = 'upper right', numpoints = 1)

plt.savefig('F0_Dvp.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <markdowncell>

# Histogram of F0 values
# ======================

# <codecell>

masked_F0 = np.ma.masked_array(all_F0, mask = np.isnan(all_F0), fill_value = 0)

plt.hist(masked_F0.compressed().ravel(), bins = 20)

plt.xlabel('F0 [Hz]')

plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (16, 12))

cmap = mpl.colors.ListedColormap(['c', 'b', 'g', 'm', 'r'])

plt.scatter(all_dVP, all_lstrain, s = all_F0 * 0.35, marker = 'o', c = all_F0, edgecolors = 'None', cmap = cmap)
cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.plot(all_dVP_np.ravel(), all_lstrain_np.ravel(), '*', ms = 15, mec = 'black', mfc = 'None', label = 'no onset')

plt.xlim(xmin = 10, xmax = 110)
plt.ylim(ymin = -10, ymax = 42)

plt.xlabel('Dvp [%]')
plt.ylabel('strain [%]')

plt.legend(loc = 'upper left', bbox_to_anchor = (0.5, 0.1), numpoints = 1)

plt.savefig('strain_Dvp_F0.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>

plt.plot(all_dVP.ravel()[chest], all_lstrain.ravel()[chest], 'g.')
plt.plot(all_dVP.ravel()[falsetto], all_lstrain.ravel()[falsetto], 'r.')

plt.plot(all_dVP_np.ravel(), all_lstrain_np.ravel(), '.', mec = 'blue', mfc = 'None')

# print "", plt.axis()

plt.xlabel('Dvp [%]')
plt.ylabel('strain [%]')
plt.title('')

plt.show()

# <markdowncell>

# Bernoulli area versus distance of vocal processes:
# --------------------------------------------------
# 
# Suggests that larger areas typically co-occur for the chest-like vibration. Smaller areas and smaller distances (adduction) typically co-occur for the high-frequency falsetto-like vibration.
# 
# There is an overlap between these.
# 
# ATTENTION
# =========
# 
# There is a sampling problem: for large areas the flow rate from the flow controller might not have been large enough to create large enough subglottal pressures for phonation onset to occur!

# <codecell>

plt.figure()
plt.plot(all_dVP.ravel()[chest], all_A.ravel()[chest], 'g.')
plt.plot(all_dVP.ravel()[falsetto], all_A.ravel()[falsetto], 'r.')
plt.plot(all_dVP_np.ravel(), all_A_np.ravel(), 'b.')

plt.xlabel('Dvp [%]')
plt.ylabel('A [a.u.]')
plt.title(r"equivalent Bernoulli area $A = Q / \sqrt{p_s}}$")

plt.figure()
plt.plot(all_lstrain.ravel()[chest], all_A.ravel()[chest], 'g.')
plt.plot(all_lstrain.ravel()[falsetto], all_A.ravel()[falsetto], 'r.')
plt.plot(all_lstrain_np.ravel(), all_A_np.ravel(), 'b.')

plt.xlabel('strain [%]')
plt.ylabel('A [a.u.]')

plt.figure()
plt.plot(all_A.ravel()[chest], all_F0.ravel()[chest], 'g.')
plt.plot(all_A.ravel()[falsetto], all_F0.ravel()[falsetto], 'r.')

plt.plot(all_A_np.ravel(), np.ones_like(all_A_np).ravel() * 50, '|', mec = 'blue', mfc = 'None')

plt.xlabel('A [a.u.]')
plt.ylabel('F0 [Hz]')

plt.show()

# <codecell>

plt.figure()
plt.scatter(all_dVP, all_A, s = all_F0 * 0.1, marker = 'o', c = all_F0, edgecolors = 'None', cmap = cmap)
cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.plot(all_dVP_np.ravel(), all_A_np.ravel(), '.', mec = 'black', mfc = 'None')

plt.xlabel('Dvp [%]')
plt.ylabel('A [a.u.]')

plt.xlim(xmax = 110)

plt.figure()
plt.scatter(all_lstrain, all_A, s = all_F0 * 0.1, marker = 'o', c = all_F0, edgecolors = 'None', cmap = cmap)
cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.plot(all_lstrain_np.ravel(), all_A_np.ravel(), '.', mec = 'black', mfc = 'None')

plt.xlabel('strain [%]')
plt.ylabel('A [a.u.]')

plt.xlim(xmin = -10, xmax = 42)

plt.show()

# <codecell>

plt.plot(all_Q.ravel()[chest], all_ps.ravel()[chest], 'g.')
plt.plot(all_Q.ravel()[falsetto], all_ps.ravel()[falsetto], 'r.')

plt.plot(all_Q_np.ravel(), all_ps_np.ravel(), 'b.')

plt.xlabel('Q [ml/s]')
plt.ylabel('ps [Pa]')
plt.title('at onset')

plt.figure()
plt.scatter(all_Q, all_ps, s = all_F0 * 0.1, marker = 'o', c = all_F0, edgecolors = 'None')
cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.plot(all_Q_np.ravel(), all_ps_np.ravel(), '.', mec = 'black', mfc = 'None')

plt.ylim(ymin = 0)

plt.xlabel('Q [ml/s]')
plt.ylabel('ps [Pa]')

plt.figure()
plt.scatter(np.log10(all_Q), np.log10(all_ps), s = all_F0 * 0.1, marker = 'o', c = all_F0, edgecolors = 'None')
cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.plot(np.log10(all_Q_np.ravel()), np.log10(all_ps_np.ravel()), '.', mec = 'black', mfc = 'None')

# plt.ylim(ymin = 0)

plt.xlabel('log10(Q [ml/s])')
plt.ylabel('log10(ps [Pa])')

plt.figure()
plt.scatter(all_Q, all_A, s = all_F0 * 0.1, marker = 'o', c = all_F0, edgecolors = 'None')
cb = plt.colorbar()
cb.set_label('frequency [Hz]')

plt.plot(all_Q_np.ravel(), all_A_np.ravel(), '.', mec = 'black', mfc = 'None')

plt.ylim(ymin = 0)

plt.xlabel('Q [ml/s]')
plt.ylabel('A [a.u.]')

plt.show()

# <codecell>

plt.plot(all_Q.ravel())
plt.plot(all_Q_np.ravel())

plt.show()

# <codecell>

plt.plot(all_ps.ravel())
plt.plot(all_ps_np.ravel())

plt.show()

# <codecell>

plt.plot(dogdata.allQ[0, :])
plt.plot(dogdata.allps[0, :], 'r')

plt.show()

# <codecell>

import numpy as np
import numpy.linalg as npl

def PCA(datamatrix, centering = False):
    """
    Performs a PCA, a Principal Component Analysis
    
    datamatrix: different variables along columns, different observations along rows
    centering: remove the mean of each column, so calculate the mean along/down each row for each column
    
    returns: eigval, norm_eigval, eigvec, projection_vec
    """
    
    Nrows, Ncols = datamatrix.shape
    
    # calculate mean along/down each row for each column
    mean_dat = np.mean(datamatrix, axis = 0)

    mean_datamatrix = np.tile( mean_dat, (Nrows, 1) )
    
    if centering:
        centered = datamatrix - mean_datamatrix
    else:
        centered = datamatrix
        
    # correlation matrix, summation over/along the rows in each column of the data matrix
    corr = np.dot(centered.T, centered)
    
    # weights (eigenvalues) and normalized eigenvectors of Hermitian or symmetric matrix
    # eigenvalues are not ordered in general
    # eigenvectors are in columns k, i.e. v[:, k]
    w, v = npl.eigh(corr)
    
    # sort from largest to smallest
    sortindex = np.argsort(w)[::-1]
    
    # sort eigenvectors
    eigen_vec = v[:, sortindex]

    # sort from largest to smallest
    eigen_val = sorted(w, reverse = True)
    # normalize eigenvalues
    norm_eigen_val = eigen_val / np.sum(eigen_val)
    
    # bi-orthogonal vectors from projection of data onto PCA directions, vectors are in columns
    projection_vec = np.dot(centered, eigen_vec)
    
    return eigen_val, norm_eigen_val, eigen_vec, projection_vec

# <codecell>

print all_F0.shape
Nall = np.prod(all_F0.shape)

# <codecell>

vardir = dict(F0 = 0, ps = 1, Q = 2, lstrain = 3, rstrain = 4, dVP = 5)

alldata = np.hstack( (all_F0.reshape((Nall, -1)), 
                      all_ps.reshape((Nall, -1)),
                      all_Q.reshape((Nall, -1)),
                      all_lstrain.reshape((Nall, -1)),
                      all_rstrain.reshape((Nall, -1)),
                      all_dVP.reshape((Nall, -1)),
                    ) )

Nallrows, Nallcols = alldata.shape

# <codecell>

# masked array to deal with missing data
ma_all = np.ma.array(alldata, mask = np.isnan(alldata))

# <codecell>

print "mean: ", ma_all.mean(axis = 0)
print "var: ", ma_all.var(axis = 0)

# <codecell>

# scale and normalize different columns because the data variables have different meanings
scaled_all = ma_all - ma_all.mean(axis = 0)

scaled_all /= np.tile(np.std(scaled_all, axis = 0), (320, 1))

# <codecell>

# make sure our data has unit variance and unit standard deviation
np.var(scaled_all, axis = 0)

# <codecell>

# array index arrays where the masked (i.e. missing) data are located
np.where(scaled_all.mask == True)

# <codecell>

rowind_mask, colind_mask = np.where(scaled_all.mask == True)
rowind_dat, colind_dat = np.where(scaled_all.mask == False)

# <codecell>

# which rows are the missing data in?
missing_data_rows = np.unique(rowind_mask)

missing_data_rows.size

# <codecell>

# remove all the masked entries and return the array as a long vector
# need to reshape
scaled_all_data = np.reshape(scaled_all.compressed(), (Nallrows - missing_data_rows.size, Nallcols))

# <codecell>

plt.matshow(scaled_all, aspect = 'auto')

plt.colorbar()

plt.show()

# <codecell>

plt.matshow(scaled_all_data, aspect = 'auto')

plt.colorbar()

plt.show()

# <codecell>

eigval, norm_eigval, eigvec, projection_vec = PCA(scaled_all_data, centering = False)

# <codecell>

neg_eigvec = eigvec.mean(axis = 0) < 0

# <codecell>

eigvec[:, neg_eigvec] *= -1
projection_vec[:, neg_eigvec] *= -1

# <codecell>

plt.figure(figsize = (14, 20))

for k in range(6):
    plt.subplot(6, 1, k + 1)
    
    plt.plot(eigvec[:, k], 'r.-')

plt.show()

# <codecell>

plt.figure(figsize = (14, 20))

for k in range(6):
    plt.subplot(6, 1, k + 1)
    
    plt.plot(projection_vec[:, k], 'b.-')

plt.show()

# <codecell>

compind = range(0, 4)

sel_eigvec = eigvec[:, compind]
sel_proj = projection_vec[:, compind]

if type(compind) is list:
    reconstruction = np.dot(sel_proj, sel_eigvec.T)
else:
    reconstruction = np.dot(sel_proj[:, np.newaxis], sel_eigvec[:, np.newaxis].T)

fullreconstruction = np.dot(projection_vec, eigvec.T)

print "mean residual: ", np.mean(fullreconstruction - scaled_all_data)
print "std of residual: ", np.std(fullreconstruction - scaled_all_data)

# <codecell>

vardir

# <codecell>

xvar = 'rstrain'
yvar = 'F0'

plt.plot(reconstruction[:, vardir[xvar]], reconstruction[:, vardir[yvar]], 'r.')

plt.xlabel(xvar)
plt.ylabel(yvar)

plt.show()

# <codecell>

print reconstruction.shape

print "mean residual: ", np.mean(reconstruction - scaled_all_data)
print "std of residual: ", np.std(reconstruction - scaled_all_data)

# <codecell>

reconst_ma = scaled_all.copy()

# <codecell>

reduced_rowind = 0

for rowind, row in enumerate(reconst_ma):

    if rowind not in missing_data_rows:
    
        reconst_ma[rowind, :] = reconstruction[reduced_rowind, :]
        reduced_rowind += 1

# <codecell>

print "mean reconstruction error: ", np.mean(reconst_ma - scaled_all)
print "std of reconstruction error: ", np.std(reconst_ma - scaled_all)

# <codecell>

plt.matshow(reconst_ma, aspect = 'auto')

plt.colorbar()

plt.show()

# <codecell>


