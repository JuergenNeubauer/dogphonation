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

xls_dir = './RLN_asymmetry/'

# <codecell>

ls -alot $xls_dir

# <codecell>

asymmetry_book = xlrd.open_workbook(filename = os.path.join(xls_dir, 'DONE 12.11.13 RLN Asymmetry.xlsx'))

# <codecell>

show_book_content(workbook = asymmetry_book)

# <codecell>

Nstimulation = 64 # 8 * 8 = (7 + 1) * (7 + 1)

# <codecell>

onsettime_samples = get_onsettime_samples(workbook = asymmetry_book, number_of_stimulations = Nstimulation)

# <codecell>

# copied here from the program posture_onset.implant...: TAconditions

nerve_conditions = ["No SLN", "SLN 1", "SLN 2", "SLN 3", "SLN 4"]

# <codecell>

compare_lists_of_nerveconditions(onsettime_samples, nerve_conditions)

# <codecell>

for nervecond in nerve_conditions:
    print len(onsettime_samples[nervecond])

# <codecell>

write_inputdata_file(filename = 'RLNasymmetry_2013_12_11_onsettime_samples.csv', 
                     nerve_conditions = nerve_conditions, 
                     onsettime_samples = onsettime_samples)

