# Telluric correction
Automatic telluric correction for ESPRESSO and NIRPS ESO pipelines. This code is described in details in Allart et al. 2022 (ADS link: https://ui.adsabs.harvard.edu/abs/2022A%26A...666A.196A/abstract). Please cite this paper when using this code.

To summarize, the goal is to have an automatic code that is able to correct astrophysical spectra from the main absorber of the Earth atmosphere. It consists on a radiative transfert model with one atmospheric layer. The produced telluric spectrum is convoled at the instrumental resolution and a least-square minimization is performed on carrefully selected lines to optimize the parameters of the model. It is currently adapted to ESPRESSO and NIRPS but will be extended to other instruments in the near future. 


## The code is fully running on python 3.9 and requires the following packages:

- numpy
- os
- astropy
- multiprocessing
- glob
- lmfit
- scpipy
- functools
- warning

Users have to clone the repository to their personal directory and make sure that the aforementionned libraries are installed.


## Structure of the repository:
- The `telluric_correction.py` script contains all the functions to perform the telluric correction. 
- The `run_telluric_correction.py` script is to run the telluric correction. **Users only have to manipulate this file.**
- The `run_telluric_correction_for_Trigger.py` is for internal consortium use.
- The `Static_model` directory containts fits files with line list parameters and the selection of lines to perform the optimization.
- The `Static_resolution` directory includes resolution maps for the different modes of NIRPS and ESPRESSO.


## The following parameters can be modified in the `run_telluric_correction.py`:
- `files_drs` is an array of files for which the telluric correction as to be performed. It is important to only provide `S2D_BLAZE_A.fits` files to perform the telluric correction.
- `instrument` can either be `'ESPRESSO'` or `'NIRPS_drs'` for now. Functions in the `telluric_correction.py` take cares of the different instrumental modes.
- `molecules` can either be `['H2O','O2']` or `['H2O,'O2','CO2','CH4']` respectively for ESPRESSO and NIRPS. These molecules are the main absorber in their respective spectral range.
- `save_path` is the output folder to save the data.
- `save_options` is an array that can contains either or all of the following keywords: `'DRS'`, `'Extended'` and `'Telluric'`. 
  - `'DRS'` creates a `S2D_BLAZE_A_TELL_CORR.fits` file where the spectrum is telluric corrected. 
  - `'Extended'` creates a `S2D_BLAZE_A_TELL_CORR_EXTENDED.fits` file that contain the uncorrected spectrum, the corrected spectrum and the telluric spectrum. 
  - `'Telluric'` creates a `S2D_BLAZE_A_TELL.fits` file with the telluric spectrum alone. 
  - As an additionnal note, the telluric corrected spectra are set to 0 when the telluric lines have an absorption larger 90% to avoid numerical effect.
- By default the telluric correction uses multiprocessing to split the files to correct on the number of cores available - 2.
