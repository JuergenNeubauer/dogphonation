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

xls_dir = './Implant_2013_10_30/'

# <codecell>

ls -alot $xls_dir/*.xls*

# <codecell>

recurrens_book = xlrd.open_workbook(filename = os.path.join(xls_dir, 'SLNvsRLN Implants 10 30 13_dc_FINAL.xls'))

vagal_book = xlrd.open_workbook(filename = os.path.join(xls_dir, 'RSLNvsRRLN.Implants.10.30.13_dc_FINAL.xls'))

# <codecell>

show_book_content(recurrens_book)

# <codecell>

show_book_content(vagal_book)

# <codecell>

# 8 * 8 = (7 + 1) * (7 + 1)

Nstimulation = 64

# <codecell>

onsettime_samples = get_onsettime_samples(workbook = recurrens_book, number_of_stimulations = Nstimulation)

# onsettime_samples = get_onsettime_samples(workbook = vagal_book, number_of_stimulations = Nstimulation)

# <codecell>

onsettime_samples

# <codecell>

nerveconditions = onsettime_samples.keys()
print nerveconditions

# <codecell>

# copied here from the program posture_onset.implant...: TAconditions

# vagal paralysis
nerve_conditions = ["GOOD M Baseline", "GOOD_No Implant", "GOOD-No Implant Repeat",
                    "GOOD Rectangle", "GOOD M Convergent", "GOOD M Divergent", "GOOD V-Shaped",
                                        "GOOD M ELRectangle", "GOOD-ELConvergent", "GOOD-ELDivergent", "GOOD ELV-shaped"]

# recurrent nerve paralysis
# nerve_conditions = ['GOOD No Implant (No L RLN)', 'GOOD Rectangle', 'GOOD Convergent', 'GOOD Divergent',
#                     'GOOD-V-shape DC', 'GOOD ELRectangle', 'GOOD ELConvergent', 'GOOD ELDivergent', 'GOOD EL-V Shaped']

# <codecell>

# compare to available cines in cine directory
!ls -alot /extra/InVivoDog/InVivoDog_2013_10_30/SLN\ versus\ RLN

!ls -alot /extra/InVivoDog/InVivoDog_2013_10_30/right\ SLN\ versus\ right\ RLN

# <codecell>

dd = OrderedDict([(nervecondition, 
                   dict(onsettime_samples = onsettime_samples[nervecondition], 
                        cinefiledir = None)) 
             for nervecondition in nerveconditions])

# <codecell>

dd['GOOD M Baseline']['cinefiledir'] = ['SLN versus RLN', 'baseline02']

# <codecell>

videosTAconditions = [["SLN versus RLN", "baseline02"],
                                  "baseline no implant", "no implant baseline repeat", 
                                  "rectangular implant", "convergent implant",
                                  "divergent implant", "V-shaped implant",
                                  "long rectangular implant", "long convergent implant",
                                  "long divergent implant", "long V-shaped implant"]

# <codecell>

combined = zip(nerve_conditions, videosTAconditions)
print combined

# <codecell>

from collections import OrderedDict

# <codecell>

OrderedDict(combined).values()

# <codecell>

dd = OrderedDict([(nervecondition, 
                   dict(onsettime_samples = onsettime_samples[nervecondition], 
                        cinefiledir = videosTAconditions[num])) 
             for num, nervecondition in enumerate(nerve_conditions)])

# <codecell>

for key in dd.keys():
    print "{}: {}".format(key, dd[key]['cinefiledir'])
    print dd[key]['onsettime_samples']

# <codecell>

# %load "/extra/InVivoDog/Elazar/code_posture_onset/posture_onset.implant_rightSLNvsrightRLN_2013_10_23.py"

# After executing %load, a new cell containing the source code will be added.
# Be sure to add the next line (with the proper path) to overwrite the file
# with you changes.
#
# %%writefile ???

# <codecell>

compare_lists_of_nerveconditions(onsettime_samples, nerve_conditions)

# <codecell>

# compare_lists_of_nerveconditions(nerve_conditions, onsettime_samples)

# <codecell>

# check that they are all the same length
for nervecond in nerve_conditions:
    print nervecond, len(onsettime_samples[nervecond])

# <codecell>

pwd

# <codecell>

help write_inputdata_file

# <codecell>

!ls -alot /extra/InVivoDog/Elazar/inputdata

# <codecell>

write_inputdata_file(filename = 'implant_rightSLNvsrightRLN_2013_10_30_onsettime_samples.csv', 
                     nerve_conditions = nerve_conditions, 
                     onsettime_samples = onsettime_samples)

# <codecell>

l = ["right SLN versus right RLN No implant_000_Wed Oct 23 2013 16 39 43.773 273.001.cine",
     "right SLN versus right RLN No implant_999_Wed Oct 23 2013 16 39 38.776 185.001.cine"]

# <codecell>

sorted(l)

# <codecell>

import time, os

# <codecell>

def datetimestamp(filename):

    fname = os.path.basename(filename)

    cleanf = os.path.splitext(fname)[0]

    # all our experiments are on Wednesdays!!!
    datestring, usec, nsec = cleanf.split('Wed')[-1].strip().split('.')

    return time.strptime(datestring, '%b %d %Y %H %M %S')

# <codecell>

sorted(l, key = datetimestamp)

# <codecell>


