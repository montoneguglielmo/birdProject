import numpy as np
import os
import gzip
from os import listdir
from os.path import isfile, isdir, join
#import matplotlib.pyplot as plt


def file_in_folder(path_name):
     
    file_wav = []

    for f in listdir(path_name):
        fl_name = join(path_name,f) 
        extent  = os.path.splitext(fl_name)[1]
        
        if isfile(fl_name) and extent  ==  '.wav' :
		file_wav.append(fl_name)

    return file_wav


def file_in_subfold(init_dir, list_out, extent_targ):
    
    for f in listdir(init_dir):
        fl_name = join(init_dir,f)

	if isdir(fl_name):
           list_out = file_in_subfold(fl_name, list_out, extent_targ)
        if isfile(fl_name):
           root = os.path.basename(fl_name)
           root = root.split('.')[0]
           extent  = os.path.splitext(fl_name)[1]
           if extent == extent_targ and not('sa' in root):
              list_out.append(fl_name)
    
    return list_out
