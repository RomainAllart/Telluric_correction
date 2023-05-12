#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14.03.2023

@author: XDumusque
"""
import os

# Set number of used threads to 1, so that numexpr used in telluric_correction.py does not interfere with the Trigger multiprocessing
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import telluric_correction as ATC

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--instrument",
    dest="instrument",
    help="Name of the instrument",
    required=True,
    choices=["ESPRESSO", "HARPS", "HARPN", "NIRPS"],
)
parser.add_argument("-f", "--file", dest="file", help="Name of the S2D_BLAZE_A file to correct for tellurics", required=True)
parser.add_argument("-mol", "--molecules", dest="molecules", help="Molecules to be considered ['H2O','O2'] for visible, ['H2O','O2','CO2','CH4'] for NIR", default=['H2O','O2'])
args = vars(parser.parse_args())

instrument = args['instrument']
molecules = args['molecules']
if instrument == 'NIRPS':
    instrument_option = 'NIRPS_drs'
    molecules = ['H2O','O2','CO2','CH4']
else:
    instrument_option = instrument
file = args['file']
save_path = '/'.join(file.split('/')[:-1])+'/'
save_options = ['DRS', 'Telluric'] # Files to be saved. Options are ['DRS', 'Extended', 'Telluric']

ATC.Run_ATC_files([file], options=[instrument_option, molecules, save_path, save_options])
