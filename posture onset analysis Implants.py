# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, sys, glob, xlrd, csv
import numpy as np

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')

# <codecell>

sys.path.append('/extra/InVivoDog/Elazar/code_posture_onset/')

# <codecell>

%run "tools posture onset analysis.py"

# <codecell>

xls_dir = './Implant_2013_10_23/'

# <codecell>

ls -alot $xls_dir

# <codecell>

recurrens_book = xlrd.open_workbook(filename = os.path.join(xls_dir, 'REVISED.SLNvsRLN.Implants.xlsx'))
vagal_book = xlrd.open_workbook(filename = os.path.join(xls_dir, 'RSLNvsRRLN.Implants.xlsx'))

# <codecell>

show_book_content(recurrens_book)

# <codecell>

show_book_content(vagal_book)

# <codecell>

Nstimulation = 64 # 8 * 8 = (7 + 1) * (7 + 1)

# <codecell>

# onsettime_samples = get_onsettime_samples(workbook = recurrens_book, number_of_stimulations = Nstimulation)

onsettime_samples = get_onsettime_samples(workbook = vagal_book, number_of_stimulations = Nstimulation)

# <codecell>

# copied here from the program posture_onset.implant...: TAconditions

nerve_conditions = ["No Implant (No L RLN)", "Rectangle", "Convergent", "Divergent", "V-Shaped",
                                        "ELRectangle", "ELConvergent", "ELDivergent", "ELV-shaped"]

# <codecell>

compare_lists_of_nerveconditions(onsettime_samples, nerve_conditions)

# <codecell>

# check that they are all the same length
for nervecond in nerve_conditions:
    print len(onsettime_samples[nervecond])

# <codecell>

write_inputdata_file(filename = 'implant_SLNvsRLN_2013_10_23_onsettime_samples.csv', 
                     nerve_conditions = nerve_conditions, 
                     onsettime_samples = onsettime_samples)

# <codecell>


