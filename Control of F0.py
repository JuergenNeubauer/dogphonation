# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

print "matplotlib: ", mpl.is_interactive()
mpl.interactive(False)
print "pyplot: ", plt.isinteractive()
plt.interactive(False)

# <codecell>

datadir = "/extra/InVivoDog/InVivoDog_2012_03_21/data LabView/SLN_trunkRLN/"

datafiles = {"TA 0": "SLN_trunkRLN_NoTA Wed Mar 21 2012 14 46 34.hdf5",
             "TA 1": "SLN_trunkRLN_ThresholdTA_condition01 Wed Mar 21 2012 14 55 08.hdf5",
             "TA 2": "SLN_trunkRLN_TA_condition02 Wed Mar 21 2012 15 01 30.hdf5",
             "TA 3": "SLN_trunkRLN_TA_condition03 Wed Mar 21 2012 15 09 20.hdf5",
             "TA 4": "SLN_trunkRLN_MaxTA_condition04 Wed Mar 21 2012 15 17 43.hdf5"}

TAconditions = sorted(datafiles.keys())

# <rawcell>

# import pandas as pd
# pd.__version__

# <rawcell>

# d = pd.read_csv("/extra/InVivoDog/Dinesh/03_21_2012_SLNvsTrunk.csv", header = 2, na_values = ['NP', 'np', '#VALUE!', '0'], index_col = 0)

# <rawcell>

# from IPython.display import HTML
# HTML(d.to_html())

# <codecell>

csvfile = csv.reader(open("/extra/InVivoDog/Dinesh/03_21_2012_SLNvsTrunk.csv", 'r'), 
                     dialect = 'excel')

# skip first three lines
for k in range(3):
    csvfile.next()

F0 = [] # onset frequency in Hz
onset = [] # onset time in milliseconds

for row in csvfile:
    F0block = row[1:6]
    onsetblock = row[8:13]

    F0block_clean = []
    onsetblock_clean = []

    for item in F0block:
        if item not in ['NP', '0', '#VALUE!']:
            F0block_clean.append(float(item))
        else:
            F0block_clean.append(np.nan)

    for item in onsetblock:
        if item not in ['NP', '0', '#VALUE!']:
            onsetblock_clean.append(float(item))
        else:
            onsetblock_clean.append(np.nan)

    F0.append(F0block_clean)
    onset.append(onsetblock_clean)

del csvfile

onset_time = onset

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
landmarkdir = "/extra/InVivoDog/python/cine/results_save"

# <codecell>

clickfiles = sorted(glob.glob(os.path.join(landmarkdir, '*.npz')))

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

TAlevels_plot = {}
SLNlevels_plot = {}
trunkRLNlevels_plot = {}

lstrain_np_plot = {}
rstrain_np_plot = {}
dVP_np_plot = {}

Nlevels = dogdata.Nlevels
# dogdata.nervenamelist
# ['left SLN', 'right SLN', 'left RLN', 'right RLN', 'left TA', 'right TA']
# we need SLN versus RLN trunk
rellevels = dogdata.a_rellevels[:, [0, 2]]

for TAnum, TAcond in enumerate(TAconditions):
    TAlevels_plot[TAcond] = np.ones((Nlevels, Nlevels)) * TAnum
    SLNlevels_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    trunkRLNlevels_plot[TAcond] = np.zeros((Nlevels, Nlevels))
    
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
        SLNlevels_plot[TAcond][SLNlevel, RLNlevel] = SLNlevel
        trunkRLNlevels_plot[TAcond][SLNlevel, RLNlevel] = RLNlevel
        
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

all_TAlevels = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_SLNlevels = np.zeros( (Nlevels, len(TAconditions), Nlevels) )
all_trunkRLNlevels = np.zeros( (Nlevels, len(TAconditions), Nlevels) )

for TAnum, TAcond in enumerate(TAconditions):
    all_TAlevels[:, TAnum, :] = TAlevels_plot[TAcond]
    all_SLNlevels[:, TAnum, :] = SLNlevels_plot[TAcond]
    all_trunkRLNlevels[:, TAnum, :] = trunkRLNlevels_plot[TAcond]
    
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

webframe("http://scitation.aip.org/content/asa/journal/jasa/129/4/10.1121/1.3552874")

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

mpl.rcParams['figure.figsize'] = (12, 8) # (10, 5)

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

# Cluster analysis
# ==================

# <codecell>

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing im(port StandardScaler

# <codecell>

allF0_masked = np.ma.masked_array(all_F0, mask = np.isnan(all_F0))
alllstrain_masked = np.ma.masked_array(all_lstrain, mask = np.isnan(all_lstrain))
allrstrain_masked = np.ma.masked_array(all_rstrain, mask = np.isnan(all_rstrain))

alldVP_masked = np.ma.masked_array(all_dVP, mask = np.isnan(all_dVP))

allps_masked = np.ma.masked_array(all_ps, mask = np.isnan(all_ps))
allQ_masked = np.ma.masked_array(all_Q, mask = np.isnan(all_Q))

# <codecell>

SLNmasked = np.ma.masked_array(all_SLNlevels, mask = np.isnan(all_F0))
TAmasked = np.ma.masked_array(all_TAlevels, mask = np.isnan(all_F0))
trunkRLNmasked = np.ma.masked_array(all_trunkRLNlevels, mask = np.isnan(all_F0))

# <codecell>

print allF0_masked.shape
print allF0_masked.compressed().shape

# <codecell>

F0ofstrain = np.vstack((alllstrain_masked.compressed(), allF0_masked.compressed())).T

# <codecell>

variables = np.vstack((alllstrain_masked.compressed(), 
                       alldVP_masked.compressed(),
                       # allps_masked.compressed(), 
                       # allQ_masked.compressed(),
                       allF0_masked.compressed())).T

# <codecell>

F0ofstimulation = np.vstack((SLNmasked.compressed(), TAmasked.compressed(), trunkRLNmasked.compressed(),
                             allF0_masked.compressed())).T

# <codecell>

F0ofstrainscaled = StandardScaler().fit_transform(F0ofstrain)

# <codecell>

F0ofstimulationscaled = StandardScaler().fit_transform(F0ofstimulation)

# <codecell>

variablesscaled = StandardScaler().fit_transform(variables)

# <codecell>

db = DBSCAN(eps = 0.4, min_samples = 10).fit(F0ofstrainscaled)

# <codecell>

db = DBSCAN(eps = 0.1, min_samples = 10).fit(F0ofstimulationscaled)

# <codecell>

db = DBSCAN(eps = 0.4, min_samples = 10).fit(variablesscaled)

# <codecell>

core_samples = db.core_sample_indices_
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# <codecell>

n_clusters_

# <codecell>

plt.close('all')

X = variables[:, [1, 2]]

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
        markeredgecolor = 'None'
        
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        x = X[index]
        if index in core_samples and k != -1:
            markersize = 14
            markeredgecolor = 'r'
        else:
            if k == -1:
                markersize = 6
                markeredgecolor = 'None'
            else:
                markersize = 8
                markeredgecolor = 'k'
                
        plt.plot(x[0], x[1], 'o', 
                 markerfacecolor = col,
                 markeredgecolor = markeredgecolor, # 'k', 
                 markersize = markersize)

plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.savefig('clusteredF0-Dvp.pdf', dpi = 100, orientation = 'landscape', bbox_inches = 'tight')
plt.show()

# <markdowncell>

# Scatter plots
# ==============
# 
# Onset frequency versus strain: 
# ------------------------------
# 
# Suggests at least two clusters of F0 values: a linear increase in a lower 'register' cluster and another linear increase above a critical strain in a higher 'register' cluster

# <codecell>

falsetto = all_F0.ravel() > 385.0
chest = all_F0.ravel() <= 385.0

# <codecell>

del fig
plt.close('all')

# <codecell>

fig, ax = plt.subplots(figsize = (12, 12)) # (16, 12))

ax.plot(all_lstrain.ravel()[chest], all_F0.ravel()[chest], 'go', 
        mec = 'None', mfc = 'green', alpha = 0.7, ms = 20, 
        label = 'chest-like cluster')

ax.plot(all_lstrain.ravel()[falsetto], all_F0.ravel()[falsetto], 'r^', 
        ms = 20, mec = 'None', mfc = 'red', alpha = 0.7,
        label = 'falsetto-like cluster')

ax.plot(all_lstrain_np.ravel(), np.ones_like(all_lstrain_np).ravel() * 20, '|', 
         ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

ax.set_xlim(xmin = -12, xmax = 42)
# ax.set_ylim(ymin = 50)

ax.set_xlabel('strain [%]')
ax.set_ylabel('F0 [Hz]')
# plt.title('onset frequency')

ax.legend(loc = 'upper left', numpoints = 1)

# <codecell>

a_chest = all_lstrain.ravel()[chest]

# a_chest.shape = (len(a_chest), 1)

a_chest = np.vstack( [ a_chest, np.ones_like(a_chest) ] ).T

slope_chest, residuals, rank, singular_val = np.linalg.lstsq(a_chest, all_F0.ravel()[chest])
print "slope_chest: ", slope_chest
print "residuals: ", residuals
print "rank: ", rank
print "singular_val: ", singular_val

# <codecell>

a_falsetto = all_lstrain.ravel()[falsetto]

# a_falsetto.shape = (len(a_falsetto), 1)

a_falsetto = np.vstack( [ a_falsetto, np.ones_like(a_falsetto) ] ).T

slope_falsetto, residuals, rank, singular_val = np.linalg.lstsq(a_falsetto, all_F0.ravel()[falsetto])
print "slope_falsetto: ", slope_falsetto
print "residuals: ", residuals
print "rank: ", rank
print "singular_val: ", singular_val

# <codecell>

ax.plot(all_lstrain.ravel()[chest], all_lstrain.ravel()[chest] * slope_chest[0] + slope_chest[1], 'k-', alpha = 0.5)
ax.plot(all_lstrain.ravel()[chest], all_lstrain.ravel()[chest] * slope_falsetto[0] + slope_falsetto[1], 'k-', alpha = 0.5)

# <codecell>

fig

# <codecell>

fig.savefig('F0_strain.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

data_rows = np.vstack( (all_F0.ravel(), all_lstrain.ravel(), all_rstrain.ravel(), all_dVP.ravel(),
                       all_ps.ravel(), all_Q.ravel(), all_A.ravel() ) )

data_rows.shape

# <codecell>

crosscorrelations = np.ma.corrcoef(np.ma.masked_array(data_rows, np.isnan(data_rows)))

# <codecell>

corrfile = open('correlations.csv', 'wt')
csvwriter = csv.writer(corrfile)

headerlist = ['', 'F0', 'lstrain', 'rstrain', 'Dvp', 'ps', 'Q', 'A']

csvwriter.writerow(headerlist)

for varname, row in zip(headerlist[1:], crosscorrelations.tolist()):
    csvwriter.writerow([varname] + row)
    
corrfile.close()

# <codecell>

plt.matshow(crosscorrelations)
plt.colorbar()
plt.show()

# <codecell>

fig, ax = plt.subplots(figsize = (12, 8)) # (16, 12))

plt.plot(all_lstrain.ravel()[chest], all_ps.ravel()[chest], 'g.', ms = 20)
plt.plot(all_lstrain.ravel()[falsetto], all_ps.ravel()[falsetto], 'r+', ms = 20, mec = 'red', mfc = 'None')

plt.plot(all_lstrain_np.ravel(), np.ones_like(all_lstrain_np).ravel() * 20, '|', 
         ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

plt.xlim(xmax = 42)

plt.xlabel('strain [%]')
plt.ylabel('ps [Pa]')
# plt.title('onset frequency')

plt.legend(loc = 'upper left', numpoints = 1)

plt.savefig('ps_strain.pdf', orientation = 'landscape',
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
# however, fundamental frequency and glottal opening might be weakly dependent on subglottal pressure
# 
# $$ u_j \propto \omega_0 \sqrt{g} $$
# 
# $$ p_s \propto \omega_0^2 g $$
# 
# $$ Q \propto u_j g $$
# 
# Bernoulli area: $$ A = \frac{Q}{\sqrt{p_s}} \propto g   $$
# 
# aerodynamic power: $$ P = p_s Q \propto u_j g^2 \omega_0^2 \propto \omega_0^3 g^{5/2} \propto \omega_0^3 A^{5/2}$$ 
# 
# plot the two sets of independent measurements against each other: $ps Q$ versus $\omega_0^3 g^{5/2}$

# <codecell>

from IPython.display import display, Math, Latex
display(Math(r'F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i k} dx'))

# <codecell>

plt.close('all')
# fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(np.log10( all_dVP.ravel()[chest]**(5/2.0) * all_F0.ravel()[chest]**3 ), 
         all_ps.ravel()[chest] * all_Q.ravel()[chest] * 1e-6, 'g.', ms = 20)

plt.plot(np.log10( all_dVP.ravel()[falsetto]**(5/2.0) * all_F0.ravel()[falsetto]**3 ), 
         all_ps.ravel()[falsetto] * all_Q.ravel()[falsetto] * 1e-6, 'r.', ms = 20)

plt.xlim(xmax = np.log10(3.5e12))

plt.xlabel('log10( $d_{VP}^{5/2} * F_0^3$ )')
plt.ylabel('$p_s * Q [W]$')
# plt.title('onset frequency')

plt.legend(loc = 'upper right', numpoints = 1)

if False:
    plt.savefig('ps_Dvp.pdf', orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)
plt.show()

# <codecell>

plt.close('all')
# fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(np.log10(all_dVP.ravel()[chest]), 
         all_Q.ravel()[chest] / np.sqrt(all_ps.ravel()[chest]), 'g.', ms = 20)

plt.plot(np.log10(all_dVP.ravel()[falsetto]), 
         all_Q.ravel()[falsetto] / np.sqrt(all_ps.ravel()[falsetto]), 'r.', ms = 20)

# plt.xlim(xmax = 2.0e7)

plt.xlabel('log10( $d_{VP}$ )')
plt.ylabel('$Q / \sqrt{p_s}$')
# plt.title('onset frequency')

plt.legend(loc = 'upper right', numpoints = 1)

if False:
    plt.savefig('ps_Dvp.pdf', orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)
plt.show()

# <codecell>

plt.close('all')

# fig, ax = plt.subplots(figsize = (16, 12))

plt.plot(np.log10( all_dVP.ravel()[chest] * all_F0.ravel()[chest]**2 ), 
         all_ps.ravel()[chest], 'g.', ms = 20)

plt.plot(np.log10( all_dVP.ravel()[falsetto] * all_F0.ravel()[falsetto]**2 ), 
         all_ps.ravel()[falsetto], 'r.', ms = 20)

# plt.plot(all_dVP_np.ravel(), np.ones_like(all_dVP_np).ravel() * 20, '|', ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

plt.xlim(xmax = np.log10(2.0e7))

plt.xlabel('log10( $d_{VP} * F_0^2$ )')
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

if False:
    plt.savefig('ps_Dvp_F0.pdf', orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>

cmap = mpl.colors.ListedColormap(['c', 'b', 'g', 'm', 'r'])

plt.scatter(all_TAlevels, all_F0, s = (all_SLNlevels + 1)**2 * 10,
            c = all_SLNlevels,
            marker = 'o',
            cmap = mpl.colors.ListedColormap(['k', 'c', 'b', 'g', 'y', 'm', 'orange', 'r']), 
            alpha = 0.6,
            edgecolors = 'None')

cb = plt.colorbar()
cb.set_label('SLN level')

plt.xlabel('TA level')
plt.ylabel('onset F0 [Hz]')

plt.xlim(xmin = -0.5, xmax = 4.5)
plt.clim(vmin = -0.5, vmax = 7.5)

plt.show()

# <codecell>

# fig, ax = plt.subplots(figsize = (16, 12))

plt.scatter(all_dVP * all_F0**2, all_ps, s = all_F0**2 * 1.0e-3, 
            marker = 'o', c = all_F0, edgecolors = 'None', cmap = cmap, alpha = 0.6)

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

plt.scatter(all_dVP * all_F0**2, all_ps, s = all_dVP**2 * 0.1, 
            marker = 'o', c = all_dVP, 
            edgecolors = 'None', cmap = cmap, alpha = 0.6)

cb = plt.colorbar()
cb.set_label('Dvp [%]')

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

plt.scatter(all_dVP * all_F0**2, all_ps, s = all_A**2 * 0.2, 
            marker = 'o', c = all_A, 
            edgecolors = 'None',
            # facecolors = 'None',
            alpha = 0.6,
            cmap = cmap)

cb = plt.colorbar()
cb.set_label('Bernoulli area [a.u.]')

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

plt.scatter(all_dVP * all_F0**2, all_ps, s = (all_lstrain - np.nanmin(all_lstrain) + 1)**2 * 0.2,
            marker = 'o', c = all_lstrain, 
            edgecolors = 'None',
            # facecolors = 'None',
            alpha = 0.6,
            cmap = cmap)

cb = plt.colorbar()
cb.set_label('left strain [%]')

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

plt.scatter(all_dVP * all_F0**2, all_ps, s = (1 + all_TAlevels) * 30.0, 
            marker = 'o', c = all_TAlevels, 
            edgecolors = 'None',
            # facecolors = 'None',
            alpha = 0.6,
            cmap = cmap)

cb = plt.colorbar()
cb.set_label('TA level')

plt.xlim(xmin = 0, # xmax = 5.5e6)
         xmax = 2.0e7)
plt.ylim(ymin = 0, ymax = 2200)

plt.clim(vmin = -0.5, vmax = 4.5)

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

plt.scatter(all_dVP * all_F0**2, all_ps, s = (1 + all_SLNlevels) * 30.0, 
            marker = 'o', c = all_SLNlevels, 
            edgecolors = 'None',
            # facecolors = 'None',
            alpha = 0.6,
            cmap = cmap)

cb = plt.colorbar()
cb.set_label('SLN level')

plt.xlim(xmin = 0, # xmax = 5.5e6)
         xmax = 2.0e7)
plt.ylim(ymin = 0, ymax = 2200)

plt.clim(vmin = -0.5, vmax = 4.5)

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

plt.scatter(all_dVP * all_F0**2, all_ps, s = (1 + all_trunkRLNlevels) * 20.0, 
            marker = 'o', c = all_trunkRLNlevels, 
            edgecolors = 'None',
            # facecolors = 'None',
            alpha = 0.6,
            cmap = cmap)

cb = plt.colorbar()
cb.set_label('trunkRLN level')

plt.xlim(xmin = 0, # xmax = 5.5e6)
         xmax = 2.0e7)
plt.ylim(ymin = 0, ymax = 2200)

plt.clim(vmin = -0.5, vmax = 4.5)

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

del fig

# <codecell>

plt.close('all')

# <codecell>

fig, ax = plt.subplots(figsize = (12, 12))

ax.plot(all_ps.ravel()[chest], all_F0.ravel()[chest], 'go', 
        mec = 'None', mfc = 'green', alpha = 0.7, ms = 20, 
        label = 'chest-like cluster')

ax.plot(all_ps.ravel()[falsetto], all_F0.ravel()[falsetto], 'r^', 
        ms = 20, mec = 'None', mfc = 'red', alpha = 0.7,
        label = 'falsetto-like cluster')

ax.plot(all_ps_np.ravel(), np.ones_like(all_ps_np).ravel() * 20, '|', 
         ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

ax.set_xlabel('ps [Pa]')
ax.set_ylabel('F0 [Hz]')

ax.legend(loc = 'upper left', bbox_to_anchor = (1200, 220), bbox_transform = ax.transData, numpoints = 1)

# <codecell>

fig

# <codecell>

fig.savefig('F0_ps.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

# <rawcell>

# falsetto = all_F0.ravel() > 385
# chest = all_F0.ravel() <= 385

# <codecell>

TAconditions

# <codecell>

chest_TA = {}
falsetto_TA = {}

for TAnum, TAcond in enumerate(TAconditions):
    chest_TA[TAnum]    = all_F0[:, TAnum, :].ravel() <= 385
    falsetto_TA[TAnum] = all_F0[:, TAnum, :].ravel() > 385

# <codecell>

colorlist = ['c', 'b', 'g', 'm', 'r', 'k']
symblist  = ['*', '+', '^', 'v', 'o', '|']

labellist = ['TA 0', 'TA 1', 'TA 2', 'TA 3', 'TA 4']

# <codecell>

del fig
plt.close('all')

# <codecell>

fig, ax = plt.subplots(figsize = (12, 12))

ax.plot(all_ps.ravel()[chest], all_F0.ravel()[chest], 'go', 
        mec = 'None', mfc = 'green', alpha = 0.8, ms = 20, 
        label = 'chest-like\ncluster', zorder = 1)

for num, (color, symbol, label) in enumerate(zip(colorlist, symblist, labellist)):

    ax.plot(all_ps[:, num, :].ravel()[falsetto_TA[num]], all_F0[:, num, :].ravel()[falsetto_TA[num]], 
            marker = symbol, label = label, color = color, mec = color,
            ms = 20, mfc = 'None', ls = 'None', alpha = 0.6)

if False:
    ax.plot(all_ps[:, 0, :].ravel()[falsetto_TA[0]], 
            all_F0[:, 0, :].ravel()[falsetto_TA[0]], 'c*', ms = 20, mec = 'k', mfc = 'None', label = 'TA 0', alpha = 0.6)
    ax.plot(all_ps[:, 1, :].ravel()[falsetto_TA[1]],
            all_F0[:, 1, :].ravel()[falsetto_TA[1]], 'b+', ms = 20, mec = 'b', mfc = 'None', label = 'TA 1', alpha = 0.6)
    ax.plot(all_ps[:, 2, :].ravel()[falsetto_TA[2]], 
            all_F0[:, 2, :].ravel()[falsetto_TA[2]], 'g^', ms = 20, mec = 'g', mfc = 'None', label = 'TA 2', alpha = 0.6)
    ax.plot(all_ps[:, 3, :].ravel()[falsetto_TA[3]],
            all_F0[:, 3, :].ravel()[falsetto_TA[3]], 'mv', ms = 20, mec = 'm', mfc = 'None', label = 'TA 3', alpha = 0.6)
    ax.plot(all_ps[:, 4, :].ravel()[falsetto_TA[4]],
            all_F0[:, 4, :].ravel()[falsetto_TA[4]], 'ro', ms = 20, mec = 'r', mfc = 'None', label = 'TA 4', alpha = 0.6)

ax.plot(all_ps_np.ravel(), np.ones_like(all_ps_np).ravel() * 50, '|', 
        ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

ax.set_xlabel('ps [Pa]')
ax.set_ylabel('F0 [Hz]')

ax.legend(loc = 'upper left', bbox_to_anchor = (1600, 450), bbox_transform = ax.transData, numpoints = 1)

# <codecell>

fig

# <codecell>

fig.savefig('F0_ps_TAcolored_alt.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

del fig
plt.close('all')

# <codecell>

# all_F0 [ SLNlevel, TAlevel, trunkRLNlevel ]

fig, ax = plt.subplots(figsize = (12, 12))

for num, (color, symbol, label) in enumerate(zip(colorlist, symblist, labellist)):
    
    ax.plot(all_lstrain[:, num, :].ravel(), all_F0[:, num, :].ravel(), 
            marker = symbol, color = color, mec = color, label = label,
            ms = 20, mfc = 'None', ls = 'None', alpha = 0.7)
    
if False:    
    ax.plot(all_lstrain[:, 0, :].ravel(), all_F0[:, 0, :].ravel(), 'c*', ms = 20, mec = 'c', mfc = 'None', label = 'TA 0', alpha = 0.7)
    ax.plot(all_lstrain[:, 1, :].ravel(), all_F0[:, 1, :].ravel(), 'b+', ms = 20, mec = 'b', mfc = 'None', label = 'TA 1', alpha = 0.7)
    ax.plot(all_lstrain[:, 2, :].ravel(), all_F0[:, 2, :].ravel(), 'g^', ms = 20, mec = 'g', mfc = 'None', label = 'TA 2', alpha = 0.7)
    ax.plot(all_lstrain[:, 3, :].ravel(), all_F0[:, 3, :].ravel(), 'mv', ms = 20, mec = 'm', mfc = 'None', label = 'TA 3', alpha = 0.7)
    ax.plot(all_lstrain[:, 4, :].ravel(), all_F0[:, 4, :].ravel(), 'ro', ms = 20, mec = 'r', mfc = 'None', label = 'TA 4', alpha = 0.7)
    
ax.plot(all_lstrain_np.ravel(), np.ones_like(all_lstrain_np).ravel() * 50, '|', ms = 20, mec = 'k', 
        mfc = 'None', label = 'no onset')

ax.set_xlim(xmin = -12, xmax = 42)

ax.set_xlabel('strain [%]')
ax.set_ylabel('F0 [Hz]')
# plt.title('onset frequency')

ax.legend(loc = 'upper left', bbox_to_anchor = (0, 1), numpoints = 1)

# <codecell>

ax.plot(all_lstrain.ravel()[chest], all_lstrain.ravel()[chest] * slope_chest[0] + slope_chest[1], 'k-', alpha = 0.5)
ax.plot(all_lstrain.ravel()[chest], all_lstrain.ravel()[chest] * slope_falsetto[0] + slope_falsetto[1], 'k-', alpha = 0.5)

# <codecell>

fig

# <codecell>

fig.savefig('F0_strain_TAcolored.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

del fig
plt.close('all')

# <codecell>

# make figure size square
fig = plt.figure(figsize = (20, 30)) # set to large number for better resolution, e.g. (20, 20)
fig.clf()

gs = mpl.gridspec.GridSpec(3, 2)
gs.update(wspace = 0.015, hspace = 0.015)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
ax_e = fig.add_subplot(gs[2, 0])
ax_f = fig.add_subplot(gs[2, 1])

# <codecell>

for a in [ax_a, ax_b, ax_c, ax_d]:
    a.set_xticklabels([])

for a in [ax_b, ax_d, ax_f]:
    a.set_yticklabels([])

# <codecell>

ax_a.set_ylabel('F0 [Hz]')
ax_c.set_ylabel('F0 [Hz]')
ax_e.set_ylabel('F0 [Hz]')

ax_e.set_xlabel('strain [%]')
ax_f.set_xlabel('strain [%]')

# <codecell>

plotlabels = ['a', 'b', 'c', 'd', 'e', 'f']
labelformat = dict(fontweight = 'bold', fontsize = 'xx-large', color = 'black', backgroundcolor = 'none', zorder = 1000)

# <codecell>

ax_a.plot(all_lstrain.ravel()[chest], all_F0.ravel()[chest], 'go', 
          mec = 'None', mfc = 'green', alpha = 0.7, ms = 20, 
          label = 'chest-like')

ax_a.plot(all_lstrain.ravel()[falsetto], all_F0.ravel()[falsetto], 'r^', 
          ms = 20, mec = 'None', mfc = 'red', alpha = 0.7,
          label = 'falsetto-like')

ax_a.plot(all_lstrain_np.ravel(), np.ones_like(all_lstrain_np).ravel() * 20, '|', 
          ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

ax_a.set_xlim(xmin = -12, xmax = 42)
# ax.set_ylim(ymin = 50)

ax_a.legend(loc = 'upper center', numpoints = 1)

# <codecell>

ax_a.plot(all_lstrain.ravel()[chest], all_lstrain.ravel()[chest] * slope_chest[0] + slope_chest[1], 'k-', alpha = 0.5)
ax_a.plot(all_lstrain.ravel()[chest], all_lstrain.ravel()[chest] * slope_falsetto[0] + slope_falsetto[1], 'k-', alpha = 0.5)

# <rawcell>

# colorlist = ['c', 'b', 'g', 'm', 'r', 'k']
# symblist  = ['*', '+', '^', 'v', 'o', '|']
# 
# labellist = ['TA 0', 'TA 1', 'TA 2', 'TA 3', 'TA 4']

# <codecell>

allaxes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]

# <codecell>

for TAlevel, a in enumerate(allaxes[1:]):

    a.plot(all_lstrain[:, TAlevel, :].ravel(), all_F0[:, TAlevel, :].ravel(),
           symblist[TAlevel], mec = colorlist[TAlevel], label = labellist[TAlevel],
           mfc = 'None', ms = 20)
    a.legend(loc = 'upper center', numpoints = 1)
    
    a.set_xlim(xmin = -12, xmax = 42)
    a.set_ylim(ymin = 0, ymax = 800)    

# <codecell>

for a in [ax_a, ax_c]:
    a.figure.canvas.draw()
    
    yticklabels = [item.get_text() for item in a.get_yticklabels()]

    yticklabels[0] = ''
    
    a.set_yticklabels(yticklabels)

# <codecell>

for num, a in enumerate(allaxes):
    labelformat.update(dict(transform = a.transData))
    a.text(-11, 757, plotlabels[num], **labelformat)

# <codecell>

fig

# <codecell>

fig.savefig('F0_strain_TAsubfigs.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

# <codecell>

# all_F0 [ SLNlevel, TAlevel, trunkRLNlevel ]

# fig, ax = plt.subplots(figsize = (16, 12))

mec_list = ['c', 'b', 'g', 'm', 'r']
symbol_list = ['*', '+', '^', 'v', 'o']

TAlevel = 4

plt.plot(all_lstrain[:, TAlevel, :].ravel(), all_F0[:, TAlevel, :].ravel(), 
         symbol_list[TAlevel], ms = 10, mec = mec_list[TAlevel], mfc = 'None', label = 'TA %d' % TAlevel)

plt.plot(all_lstrain_np.ravel(), np.ones_like(all_lstrain_np).ravel() * 50, '|', mec = 'k', mfc = 'None', label = 'no onset')

plt.xlim(xmin = -10, xmax = 42)
plt.ylim(ymin = 0, ymax = 800)

plt.xlabel('strain [%]')
plt.ylabel('F0 [Hz]')
# plt.title('onset frequency')

plt.legend(loc = 'upper left', bbox_to_anchor = (0, 1), numpoints = 1)

if True:
    plt.savefig('F0_strain_TA%d.pdf' % TAlevel, orientation = 'landscape',
                papertype = 'letter', format = 'pdf',
                bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <markdowncell>

# Onset frequency versus vocal process distance:
# ---------------------------------------------
# 
# NO separation possible

# <codecell>

del fig

# <codecell>

fig, ax = plt.subplots(figsize = (12, 12))

ax.plot(all_dVP.ravel()[chest], all_F0.ravel()[chest], 'go', 
        mec = 'None', mfc = 'green', alpha = 0.7, ms = 20, 
        label = 'chest-like cluster')

ax.plot(all_dVP.ravel()[falsetto], all_F0.ravel()[falsetto], 'r^', 
        ms = 20, mec = 'None', mfc = 'red', alpha = 0.7,
        label = 'falsetto-like cluster')

ax.plot(all_dVP_np.ravel(), np.ones_like(all_dVP_np).ravel() * 50, '|', 
        ms = 20, mec = 'blue', mfc = 'None', label = 'no onset')

ax.set_xlabel('Dvp [%]')
ax.set_ylabel('F0 [Hz]')
# plt.title('onset frequency')

ax.legend(loc = 'upper right', numpoints = 1)

# <codecell>

fig

# <codecell>

fig.savefig('F0_Dvp.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

# <markdowncell>

# Histogram of F0 values
# ======================

# <codecell>

masked_F0 = np.ma.masked_array(all_F0, mask = np.isnan(all_F0), fill_value = 0)

plt.hist(masked_F0.compressed().ravel(), bins = 20)

plt.xlabel('F0 [Hz]')

plt.show()

# <codecell>

cmap = mpl.colors.ListedColormap(['c', 'b', 'g', 'm', 'r'])

# <codecell>

del fig
plt.close('all')

# <codecell>

fig, ax = plt.subplots(figsize = (14, 12))

a_scatter = ax.scatter(all_dVP, all_lstrain, 
                       s = all_F0 * 0.3, 
                       marker = 'o', alpha = 0.6,
                       c = all_F0, edgecolors = 'None', cmap = cmap)

# <codecell>

cb = fig.colorbar(a_scatter, ax = ax)
cb.set_label('frequency [Hz]')

ax.plot(all_dVP_np.ravel(), all_lstrain_np.ravel(), '*', ms = 20, 
        alpha = 0.6, mec = 'black', mfc = 'None', label = 'no onset')

ax.set_xlim(xmin = 10, xmax = 110)
ax.set_ylim(ymin = -10, ymax = 42)

ax.set_xlabel('Dvp [%]')
ax.set_ylabel('strain [%]')

ax.legend(loc = 'upper left', bbox_to_anchor = (0.5, 0.1), numpoints = 1)

# <codecell>

fig

# <codecell>

fig.savefig('strain_Dvp_F0.pdf', orientation = 'landscape',
            papertype = 'letter', format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

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

plt.figure()
plt.scatter(all_dVP, all_A, s = 20, marker = 'o', c = all_lstrain, edgecolors = 'None', cmap = cmap)
cb = plt.colorbar()
cb.set_label('left strain [%]')

plt.plot(all_dVP_np.ravel(), all_A_np.ravel(), '.', mec = 'black', mfc = 'None')

plt.xlabel('Dvp [%]')
plt.ylabel('A [a.u.]')

plt.xlim(xmax = 110)

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
    
    datamatrix: different variables in/along columns, different observations in/along rows
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


