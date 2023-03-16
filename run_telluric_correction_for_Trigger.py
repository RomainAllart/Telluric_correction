#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14.03.2023

@author: XDumusque
"""

import argparse
import telluric_correction

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
parser.add_argument("-mol", "--molecules", dest="molecules", help="Molecules to be fitted ['H2O','O2','CO2','CH4']", default=['H2O','O2','CO2','CH4'])
args = vars(parser.parse_args())

instrument = args['instrument']
if instrument == 'NIRPS':
    instrument_option = 'NIRPS_drs'
else:
    instrument_option = instrument
file = args['file']
molecules = args['molecules']
save_path = '/'.join(file.split('/')[:-1])+'/'
save_options = ['DRS', 'Telluric'] # Files to be saved

telluric_correction.Run_ATC_files([file], options=[instrument_option, molecules, save_path, save_options])
