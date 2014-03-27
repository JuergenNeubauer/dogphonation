# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, sys, glob, xlrd, csv
import numpy as np

# <codecell>

sys.path.append('/extra/InVivoDog/python/cine/tools')

# <codecell>

def show_book_content(workbook = None):
    """
    info about workbook sheets
    """
    if not workbook:
        return
    
    print "work sheet info:"
    print "#, name, #rows, #cols"
    for casename in workbook.sheet_names():
        sheet = workbook.sheet_by_name(casename)
        print sheet.number, sheet.name, sheet.nrows, sheet.ncols

# <codecell>

def get_onsettime_samples(workbook = None, number_of_stimulations = None):
    """
    assumptions: data starts in row 4 (count starts from 0)
    and occupies columns 1 to 4 for onset, dummy, Npeaks, T
    
    returns: onsettime_samples dictionary
    """
    if not workbook:
        return None
    if not number_of_stimulations:
        return None
    
    onsettime_samples = {}

    for casename in workbook.sheet_names():
        sheet = workbook.sheet_by_name(casename)
    
        onset_list = [] # onset time in samples (sampling rate: 50 KHz)
    
        for rownum in range(number_of_stimulations):
            onset, dummy, Npeaks, T = sheet.row_values(rownum + 4, start_colx = 1, end_colx = 5)
        
            if Npeaks != 4:
                print "casename: ", casename
                print "rownum: ", rownum
                print "onset, dummy, Npeaks, T: ", onset, dummy, Npeaks, T
                print
                sys.stdout.flush()
                raise ValueError("Npeaks is NOT 4")
        
            try:
                if onset.strip() in ['NP', '']:
                    # onset_list.append(np.nan)
                    onset_list.append(str(onset))
            except:
                onset_list.append(int(onset))
    
        onsettime_samples[str(casename)] = onset_list
        
    return onsettime_samples

# <codecell>

def compare_lists_of_nerveconditions(onsettime_samples, nerve_conditions):
    """
    compare lists of nerve conditions: e.g. onsettime_samples with nerve_conditions
    """
    setA = set(onsettime_samples.keys())
    setB = set(nerve_conditions)
    
    d = setA.difference(setB)
    
    if len(d) == 0:
        print 'no difference'
    else:
        not_in_A = []
        not_in_B = []
        for item in d:
            if item in setA:
                not_in_B.append(item)
            if item in setB:
                not_in_A.append(item)
                
            if not_in_B:
                print 'elements not in nerve_conditions: ', not_in_B
            if not_in_A:
                print 'elements not in onsettime_samples: ', not_in_A

# <codecell>

def write_inputdata_file(filename = None, nerve_conditions = None, onsettime_samples = None):
    """
    write a csv file in /extra/InVivoDog/Elazar/inputdata
    """
    if not filename:
        print "filename not valid"
        return
    if not nerve_conditions:
        print "no nerve_conditions given"
        return
    if not onsettime_samples:
        print "no onsettime_samples given"
        return
    
    a_time_samples = np.array([onsettime_samples[nervcond] for nervcond in nerve_conditions]).T
    
    outfile = open(os.path.join('/extra/InVivoDog/Elazar/inputdata', filename),
                   'wt')
    
    csv_writer = csv.writer(outfile)
    
    csv_writer.writerow(['File #'] + nerve_conditions)
    
    for num, row in enumerate(a_time_samples):
        filenum = num + 1
        
        csv_writer.writerow([filenum] + list(row))
        
    outfile.close()

# <codecell>

def datetimestamp(cinefilename):
    """
    extract the date-time-stamp from the cine file name
    
    can be used to properly sort cine file names which have the following naming convention:
    
    right SLN versus right RLN No implant_000_Wed Oct 23 2013 16 39 43.773 273.001.cine
    right SLN versus right RLN No implant_999_Wed Oct 23 2013 16 39 38.776 185.001.cine
    """

    fname = os.path.basename(filename)

    cleanf = os.path.splitext(fname)[0]

    # all our experiments are on Wednesdays!!!
    datestring, usec, nsec = cleanf.split('Wed')[-1].strip().split('.')

    return time.strptime(datestring, '%b %d %Y %H %M %S')

