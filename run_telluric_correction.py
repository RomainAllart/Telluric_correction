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


files_nirps_drs=glob.glob('Telluric_corrected_files/r.NIRPS.*S2D_BLAZE_A.fits')
files_nirps_drs.sort()

Save_path='Telluric_corrected_files/'

if __name__ == "__main__":
    ATC.multiprocessing_ATC(files_nirps_drs,['NIRPS_drs',['H2O','O2','CO2','CH4'],Save_path],nthreads=6)

