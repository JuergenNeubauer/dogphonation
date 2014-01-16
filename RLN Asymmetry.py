# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Create graphs evaluating phase and mucosal asymmetry vs stimulation levels of RLN and SLN in the 4/4/12 RLN Asymmetry.
# 
# Tables with the phase and mucosal asymmetries on Tabs 6 & 7.
# 
# Y:\Chhetri_R01_Projects\RLN asymmetry
# 
# xls file: RLN Asymmetry_4.4.12_FINAL

# <codecell>

import sys, glob, os
import numpy as np
import xlrd

# <codecell>

%matplotlib inline

# <codecell>

%config InlineBackend.close_figures = False
%config InlineBackend

# <codecell>

import matplotlib as mpl
# mpl.use('module://IPython.zmq.pylab.backend_inline')
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['image.cmap'] = 'jet'

# <codecell>

mpl.rcParams['savefig.dpi'] = 100

# <codecell>

excel_dir = "/extra/InVivoDog/Dinesh/RLN_asymmetry/"

# <codecell>

!ls -alot $excel_dir

# <codecell>

glob.glob(os.path.join(excel_dir, '*.xlsx'))

# <codecell>

%run "tools posture onset analysis.py"

# <codecell>

# excel_filename = os.path.join(excel_dir, 'RLN Asymmetry_4.4.12_FINAL.xlsx')
excel_filename = os.path.join(excel_dir, 'DONE 12.11.13 RLN Asymmetry.xlsx')
book = xlrd.open_workbook(filename = excel_filename)

# <codecell>

print 'sheets: ', book.sheet_names()
print
for sheet in book.sheets():
    print "%d: %s: %d, %d" % (sheet.number, sheet.name, sheet.nrows, sheet.ncols)

# <codecell>

s0 = book.sheet_by_name('No SLN')

s0.row_values(3).index('Phase (L/R)')

# <codecell>

Nstimulations = 64 # (7 + 1)**2

# <codecell>

mucosal = []

for slncondition in book.sheet_names():
    print slncondition
    sheet = book.sheet_by_name(slncondition)
    
    p = sheet.col_values(colx = 8, start_rowx = 4, end_rowx = 4 + Nstimulations)
    
    mucosal.append(p)

# <codecell>

sheet = book.sheet_by_name('SLN 3')

sheet.row_values(3).index('L Level')

# <codecell>

leftRLN = sheet.col_values(colx = 10 + 0, start_rowx = 4, end_rowx = 4 + Nstimulations)
rightRLN = sheet.col_values(colx = 10 + 1, start_rowx = 4, end_rowx = 4 + Nstimulations)

# <codecell>

RLNnames = ['left RLN', 'right RLN']
SLNlevelnames = ['SLN 0', 'SLN 1', 'SLN 2', 'SLN 3', 'SLN 4']

dt = np.dtype({'names': RLNnames + SLNlevelnames,
               'formats': [np.float16] * len(RLNnames) + ['S3'] * len(SLNlevelnames)})
print "datatype: ", dt

# <codecell>

a_mucosal = np.empty((Nstimulations, 1), dtype = dt)

for SLNnum, SLNlevel in enumerate(SLNlevelnames):
    a_mucosal[SLNlevel] = np.array(mucosal[SLNnum]).reshape(-1, 1)
    
a_mucosal['left RLN'] = np.array(leftRLN).reshape(-1, 1)
a_mucosal['right RLN'] = np.array(rightRLN).reshape(-1, 1)

a_mucosal.sort(axis = 0, order = RLNnames)

a_mucosal.shape = (8, -1)

# <codecell>

for RLNname in RLNnames:
    print RLNname
    print a_mucosal[RLNname]

print 'SLN 2'
print a_mucosal['SLN 2']

# <codecell>

for SLNlevel in SLNlevelnames:
    SLNcond = a_mucosal[SLNlevel]
    SLNcond[SLNcond == 'L'] = -1.0
    SLNcond[SLNcond == 'S'] = 0.0
    SLNcond[SLNcond == 'R'] = 1.0
    SLNcond[SLNcond == 'n/a'] = np.nan
    SLNcond[SLNcond == ''] = np.nan
    SLNcond[SLNcond == '?'] = np.nan

print a_mucosal['SLN 2']

# <codecell>

plt.imshow(a_mucosal['left RLN'])
plt.title('left RLN')
plt.xlabel('right RLN')
plt.ylabel('left RLN')
plt.show()

# <codecell>

SLNlevel = 'SLN 0'

for SLNlevel in SLNlevelnames:
    plt.close('all')
    
    plt.imshow(a_mucosal[SLNlevel].astype(np.float))
    plt.plot([0, 7], [0, 7], 'k-')
    
    plt.title(SLNlevel)
    
    plt.xlabel('right RLN')
    plt.ylabel('left RLN')
    
    plt.xlim(xmin = -0.5, xmax = 7.5)
    plt.ylim(plt.xlim())
    
    plt.savefig(SLNlevel + '.pdf', dpi = 300, orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)

# <markdowncell>

# The following code is for the older RLN asymmetry data
# ------------------------------------------------------
# 
# from:
# 
# excel_filename = os.path.join(excel_dir, 'RLN Asymmetry_4.4.12_FINAL.xlsx')

# <codecell>

ls -alot $excel_dir

# <codecell>

excel_filename = os.path.join(excel_dir, 'RLN Asymmetry_4.4.12_FINAL.xlsx')
# excel_filename = os.path.join(excel_dir, 'DONE 12.11.13 RLN Asymmetry.xlsx')
book = xlrd.open_workbook(filename = excel_filename)

# <codecell>

sheet_phase = book.sheet_by_name('Phase Asymmetries')
sheet_mucosal = book.sheet_by_name('Mucosal Asymmetries')

# <codecell>

print sheet_phase.row(2)
print sheet_phase.row_values(2)

# <codecell>

sheet_phase.row_values(2, start_colx = 1)

# <codecell>

RLNnames = ['left RLN', 'right RLN']
SLNlevelnames = ['SLN 0', 'SLN 1', 'SLN 2', 'SLN 3', 'SLN 4']

dt = np.dtype({'names': RLNnames + SLNlevelnames,
               'formats': [np.float16] * len(RLNnames) + ['S3'] * len(SLNlevelnames)})
print "datatype: ", dt

# <codecell>

Nstimulations = 64 # sheet_phase.nrows - 3

phase = np.empty((Nstimulations, 1), dtype = dt)
mucosal = np.empty((Nstimulations, 1), dtype = dt)

# <codecell>

for colnum, RLNname in enumerate(RLNnames):
    # first column is empty
    phase[RLNname] = np.array(sheet_phase.col_values(1 + colnum, start_rowx = 3)).reshape(-1, 1)
    mucosal[RLNname] = np.array(sheet_mucosal.col_values(1 + colnum, start_rowx = 3)).reshape(-1, 1)
    
for colnum, SLNlevel in enumerate(SLNlevelnames):
    col_phase = [item.lower() for item in sheet_phase.col_values(3 + colnum, start_rowx = 3)]
    phase[SLNlevel] = np.array(col_phase).reshape(-1, 1)
    
    col_mucosal = [item.lower() for item in sheet_mucosal.col_values(3 + colnum, start_rowx = 3)]
    mucosal[SLNlevel] = np.array(col_mucosal).reshape(-1, 1)    

# <codecell>

# in-place sort
phase.sort(axis = 0, order = RLNnames)
mucosal.sort(axis = 0, order = RLNnames)

phase.shape = (8, -1)
mucosal.shape = (8, -1)

# <codecell>

print "left RLN"
print phase['left RLN']
print "right RLN"
print phase['right RLN']
print "phase: SLN 1"
print phase['SLN 1']

print "mucosal: SLN 1"
print mucosal['SLN 1']

# <codecell>

imshow_opts = dict(origin = 'lower', aspect = 'equal')

plt.subplot(1,2,1)
plt.imshow(phase['left RLN'], **imshow_opts)
plt.xlabel('right RLN')
plt.ylabel('left RLN')
plt.title('left RLN')

plt.subplot(1,2,2)
plt.imshow(phase['right RLN'], **imshow_opts)
plt.title('right RLN')

# plt.show()

# <codecell>

cmap = mpl.colors.ListedColormap(['b', 'g', 'r'])

# <codecell>

for a in [phase, mucosal]:
    for SLNlevel in SLNlevelnames:
        SLNcond = a[SLNlevel]
        SLNcond[SLNcond == 'l'] = -1
        SLNcond[SLNcond == 's'] = 0
        SLNcond[SLNcond == 'r'] = 1
        SLNcond[SLNcond == 'n/a'] = np.nan

print phase['SLN 1']
print mucosal['SLN 1']

# <codecell>

phase['SLN 1'].astype(np.float)

# <codecell>

import matplotlib.gridspec as gridspec

nrows = 2
ncols = 5

gs = gridspec.GridSpec(nrows, ncols, wspace = 0.05, hspace = 0.05)

fig = plt.figure(figsize = (12, 8))
axgrid = np.array([fig.add_subplot(spec) for spec in gs])
axgrid.shape = (nrows, ncols)
axgrid = np.flipud(axgrid).T

# test numbering of grid
testing = False
text_template = 'grid: %3d\nx = %2d, y = %2d'

for g in axgrid.ravel():
    g.grid(False)
    
    spec = g.get_subplotspec()
    gnum = spec.get_geometry()[-2]
    
    g.xind = gnum % ncols
    g.yind = (nrows - 1) - gnum / ncols
    
    if testing:
        text_format = dict(transform = g.transAxes, # g.transData, # axframe.axes.transData,
                           fontweight = 'bold', fontsize = 'large',
                           color = 'red', backgroundcolor = 'none')
    
        text = g.text(0.2, 0.4,
                      text_template % (gnum, g.xind, g.yind),
                      **text_format)
    
if testing:    
    # for example, plot directly into grid[2,0]
    text_format = dict(transform = axgrid[2,0].transAxes, # g.transData, # axframe.axes.transData,
                       fontweight = 'bold', fontsize = 'large', 
                       color = 'black', backgroundcolor = 'yellow')
            
    axgrid[2, 0].text(0.1, 0.1, 'coord: (2,0)', **text_format)

# <codecell>

for SLNnum, SLNlevel in enumerate(SLNlevelnames):
    # plot phase asymmetry
    a = axgrid[SLNnum, 1]
    a.imshow(phase[SLNlevel].astype(np.float), cmap = cmap, **imshow_opts)
    a_axis = a.axis()
    a.plot([0, 7], [0, 7], 'k--', zorder = 1000)
    a.axis(a_axis)
    a.title.set_text(SLNlevel)
    
    a.set_xticklabels('')
    if SLNlevel in ['SLN 0']:
        a.set_ylabel('left RLN')
        a.text(-0.8, 0.5, 'phase', transform = a.transAxes, rotation = 0, color = 'red', fontweight = 'bold')
    else:
        a.set_yticklabels('')
    
    # plot mucosal wave asymmetry: asymmetry in appearance???
    a = axgrid[SLNnum, 0]
    a.imshow(mucosal[SLNlevel].astype(np.float), cmap = cmap, **imshow_opts)
    a_axis = a.axis()
    a.plot([0, 7], [0, 7], 'k--', zorder = 1000)
    a.axis(a_axis)
    
    if SLNlevel in ['SLN 0']:
        a.set_xlabel('right RLN')
        a.set_ylabel('left RLN')
        a.text(-0.8, 0.5, 'mucosal\nwave', transform = a.transAxes, color = 'red', fontweight = 'bold')
    else:
        a.set_yticklabels('')

gs.update(bottom = 0.4)

plt.savefig('phase_mucosal_asymmetry.pdf', format = 'pdf', papertype = 'letter', dpi = 300,
            orientation = 'landscape', bbox_inches = 'tight', pad_inches = 0.1)

plt.show()

# <codecell>


