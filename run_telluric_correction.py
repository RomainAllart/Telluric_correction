#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:49:49 2023

@author: romainallart
"""


###################
####    DATA   ####
###################
import glob
import telluric_correction as ATC
import multiprocessing


files_drs=glob.glob('Telluric_corrected_files/r.NIRPS.*S2D_BLAZE_A.fits')
files_drs.sort()

instrument = 'NIRPS_drs'
molecules = ['H2O','O2','CO2','CH4'] # molecules to consider
save_path='Telluric_corrected_files/' # directory to save files
save_options = ['DRS', 'Extended', 'Telluric'] # Files to be saved

if __name__ == "__main__":
    ATC.multiprocessing_ATC(files_drs,[instrument, molecules, save_path, save_options],nthreads=max(1,multiprocessing.cpu_count()-2))

