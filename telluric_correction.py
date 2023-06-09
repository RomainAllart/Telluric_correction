#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:12:07 2020

@author: allartromain
"""

import numpy as np
import os
import sys
from astropy.io import fits
from multiprocessing import Pool
from lmfit import minimize, Parameters, report_fit, fit_report
import time
from scipy.special import wofz
from scipy.interpolate import interp1d
from lmfit import Model
from functools import partial
import warnings
# import fast_convolution as fast_conv
# from scipy import constants
import numexpr as ne
ne.set_vml_num_threads(1)
import logging

warnings.filterwarnings("ignore")

logger = logging.getLogger('Telluric')
# Tu peux mettre le nom que tu veux
logger.setLevel(logging.INFO)
# C'est le niveau de log min que tu veux voir à l'écran

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Ca va te formatter tous les logs avec la date, l’heure, le niveau de log et le message
handler.setFormatter(formatter)
logger.addHandler(handler)


def NIRPS_resolution_temp(ins_mode):
    if ins_mode=='HA':
        return 88000
    elif ins_mode =='HE':
        return 75000
    else:
        return None

def create_folder(name):
    try:
        #os.mkdir(name)
        os.system('mkdir -p '+name)
    except OSError:
        pass
    return

def ind_array(array,values):
    """Search closest indice from a given value in array"""
    if np.isscalar(values) == False:
        indice_list = []
        for value in range(len(values)):
            indice_list.append(np.argmin(np.abs(array - values[value])))
        return np.array(indice_list)
    else:
        return np.argmin(np.abs(array - values))

def fits2wave(file_or_header):
    info = """
        Provide a fits header or a fits file
        and get the corresponding wavelength
        grid from the header.
​
        Usage :
          wave = fits2wave(hdr)
                  or
          wave = fits2wave('my_e2ds.fits')
​
        Output has the same size as the input
        grid. This is derived from NAXIS
        values in the header
    """

    # check that we have either a fits file or an astropy header
    if type(file_or_header) == str:
        hdr = fits.getheader(file_or_header)
    elif str(type(file_or_header)) == "<class 'astropy.io.fits.header.Header'>":
        hdr = file_or_header
    else:
        print()
        print('~~~~ wrong type of input ~~~~')
        print()

        print(info)
        return []

    # get the keys with the wavelength polynomials
    wave_hdr = hdr['WAVE0*']
    # concatenate into a numpy array
    wave_poly = np.array([wave_hdr[i] for i in range(len(wave_hdr))])

    # get the number of orders
    nord = hdr['WAVEORDN']

    # get the per-order wavelength solution
    wave_poly = wave_poly.reshape(nord, len(wave_poly) // nord)

    # get the length of each order (normally that's 4088 pix)
    try:
        npix = hdr['NAXIS1']
    except:
        npix = fits.open(file_or_header)[1].header['NAXIS1']


    # project polynomial coefficiels
    wavesol = [np.polyval(wave_poly[i][::-1], np.arange(npix)) for i in range(nord)]
    
    # return wave grid
    return np.array(wavesol)

def voigt(x, HWHM=1, gamma=1, center=0):
    """Compute voigt profile"""
    sigma = HWHM / (np.sqrt(2.*np.log(2)))
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2.))
    V = wofz(z) / (np.sqrt(2. * np.pi) * sigma)
    return np.real(V)




def supergauss(x, ew, expo):
    """
    :param x:
    :param fwhm:
    :param expo:
    :return:

    0.5 = exp(-0.5*((fwhm/2)/ew)^expo)
    np.log(0.5) = -0.5*((fwhm/2)/ew)^expo
    2*np.log(2) = ((fwhm/2)/ew)^expo
    """

    # ew = (fwhm/2)/(2*np.log(2))**(1/expo)
    # return np.exp(-0.5*np.abs(x/ew)**expo)
    
    #Expression d'Etienne original mais avec une erreur
    # str_exp = 'exp(-0.5 * abs(x / (fwhm/2)/(2*log(2))**(1/expo)) ** expo)' 
    
    #Expression corrige brute
    # str_exp = 'exp(-0.5 * abs(x / ((fwhm/2)/(2*log(2))**(1/expo))) ** expo)'
    
    #Expression corrige avec la logique de depary
    str_exp = 'exp(-0.5 * abs(x / ew) ** expo)'

    return ne.evaluate(str_exp)

def compute_CCF(l,spe,mask_ll,mask_W,RV_table,bin_width,normalize,excluded_ll,excluded_range,instrument='HARPS',rescale=0):

    """
    Compute the Cross Correlation Function for a given set of radial velocities and an 'observed spectrum'.

    The CCF is computed with an arbitrary binary mask rather than with a complete theoretical model. This apporach allows to build an
    average profile of the lines naturally because of the binarity of the mask (RV_table, CCF). It is average, because for each fixed
    RV you sum on all the lines in the line list.
    In the other approach (a la Snellen), the interpretation of the shape of (RV_table, CCF) is more complicated since it depends on
    the shape of the lines in the models we use for the cross-correlation.
    """

    # l, spe: spe(l) is the observed spectrum at wavelength l;
    # mask_ll: the line list (wavelength);
    # mask_W: weight of the single line (Pepe+, 2002). In transmission spectroscopy this is delicate, and thus set to 1 for every line.
    # RV_table: the table of radial velocities of the star behind the planet. Corresponds to a shift in the mask. Max CCF = true RV (theoretically)
    #           for HARPS, use -20/+20 km/sec in the planetary rest frame.
    # bin_width: the width of the mask bin in radial velocity space (= 1 pixel HARPS; too big: bending lines you lose information; too small: ????)
    #            Rs = 115000; dv_FWHM = c/Rs ~ 2.607 km/sec; dv_pix = dv_FWHM/sampling = 2.607/3.4 km/sec = 0.76673 km/sec
    # normalize: just for telluric correction. 1 = True
    # excluded_ll: used to exclude specific contaminating wavelengths; e.g. O2
    # excluded_range: same as before but with a range in the wavelength space
    #                 NOT SURE ABOUT THIS. From the code, it seems that a line in the 'excluded_ll' is rejected only if it falls inside this region.
    #                 I guess you may want to exclude the lines for a certain element only in certain regions of the spectrum, while you may want to
    #                 include them in others.
    # instrument: only important for HARPS, that has a gap between the CCDs
    # rescale: ????

    # This is just the spacing between the wavelengths, and a nice way to that by the way.
    dl = l[1:]-l[:-1]
    dl = np.concatenate((dl,np.array([dl[-1]])))

    # Centers of the wavelength bins (there is a reason for this, I don't remember which).
    l2 = l-dl/2.

    # CCF contains the CCF for each radial velocity (thus has the size of RV_table)
    CCF = np.zeros(len(RV_table),'d')
    rejected = 0

    # Cycle on the line list
    for j in range(len(mask_ll)):
        dCCF = np.zeros(len(RV_table),'d')
            # Jump the gap between the CCDs if the data comes from HARPS
        if (instrument == 'HARPS') & (mask_ll[j] > 5299.) & (mask_ll[j] < 5342.):
            continue
            # 2.99792458e5 = c in km/sec
            # Exclude polluters.
        #if abs(excluded_ll-mask_ll[j]).min() < (mask_ll[j]*excluded_range/2.99792458e5):
        if np.min(abs(excluded_ll-mask_ll[j])) < (mask_ll[j]*excluded_range/2.99792458e5):

            rejected = rejected+1
            continue
        # Compute the width of the mask in the wavelength space.
        # lambda/delta_lambda = c/delta_v ---> delta_lambda = lambda * delta_v /c
        mask_width = mask_ll[j]*bin_width/2.99792458e5
            # For each line in the line list, cycle on the RVs and compute the contribution to the CCF.
        for k in range(len(RV_table)):
            # The planet moves with respect to the star. It absorbs in its rest frame, thus it is the 'observer' in this case -> Doppler formula with the plus.
            # Check: positive velocity is away from the star; negative is towards the star. In fact:
            # When the planet moves towards th star, RV < 0, thus the wavelength at which is absorbs is bluer than the original.
            lstart = mask_ll[j]*(1.+RV_table[k]/2.99792458e5)-mask_width/2.
            lstop  = mask_ll[j]*(1.+RV_table[k]/2.99792458e5)+mask_width/2.

                # index1 is the index of the element in l2 such that if I insert lstart on its left the order is preserved. Thus, it is the index of the first wavelength
                # bin contained in the mask.
                # index2 is the index of the first wavelength bin not completely covered by the mask.
            index1 = np.searchsorted(l2,lstart)
            index2 = np.searchsorted(l2,lstop)

                # First term: all the bins completely contained in the mask. Second term: I have to add a piece of bin on the left of index1 which has been excluded in the
                # first term. Third term: the last bin contributes to the CCF only partially, remove the excess.
            dCCF[k] = sum(spe[index1:index2]) + spe[index1-1]*(l2[index1]-lstart)/dl[index1-1] - spe[index2-1]*(l2[index2]-lstop)/dl[index2-1]

        if normalize == 1:
            index = len(RV_table)//5
#               slope = (dCCF[-index:].mean()-dCCF[:index].mean())/(RV_table[-index:].mean()-RV_table[:index].mean())
#               c = dCCF[:index].mean() + slope*(RV_table-RV_table[:index].mean())
            c = np.concatenate((dCCF[:index],dCCF[-index:])).mean()
            if (dCCF.min() <= 0.) | (c <= 0.): continue
            dCCF = np.log(dCCF/c)
        if rescale == 1:
            index = np.searchsorted(l,mask_ll[j])
            dCCF  = dCCF*dl[index]/mask_width

        # Sum the weighted contribution of each line to the CCF.
        CCF = CCF + mask_W[j]*dCCF
    if excluded_range > 0.:
        print('Rejected %i/%i lines in excluded wavelength range'%(rejected,len(mask_ll)))
    if rescale==1:
        CCF = CCF/len(mask_ll)
    return CCF

def fit_telluric_model(Parameters,rv,data=['wave','flux','database','qt_list','lines_position_fit','map_resolution','N_x','M_mol','molecule','instrument']):
    """ Function to fit and apply the best telluric absorption model around the 10 strongest lines for a list of orders given as input"""

	##################################
	####    Constants and Inputs  ####
	##################################

    R     = 3000000 # model resolution

    c     = 299792458.0                 # [m*s^-1]
    c_km_s = c/1000
    NA    = 6.02214086e+23              # [mol^-1]
    k_b   = 1.3806503e-23               # [J*K^-1] = [kg*m^2*s^-2]
    c2    = 1.43880285                  # c*h/k in [cm*K]

    parvals     = Parameters.valuesdict()
    temp        = parvals['Temperature']
    P_0         = parvals['Pressure_ground']
    PWV_airmass = parvals['PWV_w_airmass']

    Wave_input         = data[0]
    Flux_input         = data[1]
    hitran_database    = data[2]
    qt_used            = data[3]['Qt'][np.where(data[3]['Temperature']==round(temp))[0]][0]
    lines_position_fit = data[4]
    Resolution_map     = data[5]
    N_x                = data[6]
    M_mol              = data[7]
    molecule           = data[8]
    instrument         = data[9] #to be removed for Danuta



    ######################################################################
    ####    Identify all orders/slices where lines have to be fitted  ####
    ######################################################################

    wave_number_hitran_rest_fit   = hitran_database['wave_number']
    intensity_hitran_fit          = hitran_database['Intensity']
    gamma_air_fit                 = hitran_database['gamma_air']
    n_air_fit                     = hitran_database['N']
    delta_air_fit                 = hitran_database['delta']
    Epp_fit                       = hitran_database['Epp']

    wave_number_hitran_fit_scaled     = wave_number_hitran_rest_fit + delta_air_fit * P_0
    intensity_hitran_fit_scaled       = intensity_hitran_fit * (174.5813 / qt_used) * (np.exp(- c2 * Epp_fit / temp)) / (np.exp(- c2 * Epp_fit / 296.0)) * (1. - np.exp(- c2 * wave_number_hitran_rest_fit / temp)) / (1. - np.exp(- c2 * wave_number_hitran_rest_fit / 296.0))
    hwhm_lorentzian_hitran_fit_scaled = (296.0 / temp) ** n_air_fit * gamma_air_fit * P_0  # assuming P_mol=0
    if molecule != 'H2O':
        hwhm_gaussian_hitran_fit_scaled = wave_number_hitran_rest_fit / c * np.sqrt(2. * NA * k_b * temp * np.log(2) / (10 ** -3 * M_mol))

    Orders=[]
    for i in range(len(Wave_input)):
        if any((10**8/(Wave_input[i][-1] - 4*40./300000*np.mean(Wave_input[i])) < lines_position_fit['CCF_lines_position_wavenumber']) &  (lines_position_fit['CCF_lines_position_wavenumber'] < 10**8/(Wave_input[i][0] + 4*40./300000*np.mean(Wave_input[i])))):
            Orders.append(i)

    #########################################################
    ####    Create model for the selected orders/slices  ####
    #########################################################
    ccf_uncorr_ord     = []
    ccf_model_conv_ord = []

    for Ord in range(len(Orders)):
        """loop on the selected orders"""
        Order=Orders[Ord]

        data_wave = Wave_input[Order]
        data_flux = Flux_input[Order]
        step      = data_wave[len(data_wave)//2]/R
        born      = data_wave[len(data_wave)//2]*40/300000


        wave                     = np.arange(data_wave[0]-born,data_wave[-1]+born,step) #create synthetic spectrum on a broader range
        wave_wavenumber          = 1./(wave*10**-8)
        wave_wavenumber_selected = 1./(np.arange(data_wave[0]+4.*born,data_wave[-1]-4*born,np.mean(np.diff(data_wave)))*10**-8) # selects lines in a narrower range


        intensity_hitran            = intensity_hitran_fit_scaled[       (wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
        wave_number_hitran_P0       = wave_number_hitran_fit_scaled[     (wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
        hwhm_lorentzian_hitran_P0   = hwhm_lorentzian_hitran_fit_scaled[ (wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
        if molecule != 'H2O':
            hwhm_gaussian_hitran    = hwhm_gaussian_hitran_fit_scaled[   (wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]

    	####################################
    	####    Compute telluric model  ####
    	####################################

        spectrum=np.zeros(len(wave))

        for i in range(len(wave_number_hitran_P0)):

            if molecule == 'H2O': #lorentzian profile
                line_profile = 1. / np.pi * hwhm_lorentzian_hitran_P0[i] / ( hwhm_lorentzian_hitran_P0[i]**2. + ( wave_wavenumber - wave_number_hitran_P0[i] )**2. )
            else:
                line_profile = voigt(wave_wavenumber, HWHM=hwhm_gaussian_hitran[i], gamma=hwhm_lorentzian_hitran_P0[i], center=wave_number_hitran_P0[i])

            spectrum = np.add(spectrum, intensity_hitran[i] * line_profile)

        telluric_spectrum = np.exp( - PWV_airmass * N_x * spectrum )


    	######################################
    	####    Convolution and binning   ####
    	######################################
        
        if instrument == 'SPIRou_as_NIRPS': #to be removed for Danuta
            data_wave_temp = wave
            beta=2.24 #top sqaure for spirou
            resolution_pixel    = 60000
            fwhm_angstrom_pixel = wave[len(wave)//2]/resolution_pixel
            sigma_angstrom      = fwhm_angstrom_pixel / 2.3548  
            wave_psf            = wave[len(wave)//2-int(np.ceil(4*sigma_angstrom/step)):len(wave)//2+int(np.ceil(4*sigma_angstrom/step))+1]
            gaussian_psf        = np.multiply( np.array(1. / (sigma_angstrom * np.sqrt( 2. * np.pi ) ) ) , np.exp( - np.true_divide( np.abs( wave_psf - wave[len(wave)//2] ) **beta , np.array( 2. * sigma_angstrom ** beta ) ) ) )
            gaussian_psf_norm   = np.true_divide(gaussian_psf,np.array(np.sum(gaussian_psf)))
            telluric_spectrum_conv = np.convolve(telluric_spectrum,gaussian_psf_norm,mode='same')             
        else:


            spectrum = telluric_spectrum#np.ones(len(telluric_spectrum)) #telluric_spectrum
            f=interp1d(data_wave,Resolution_map[Order],kind='cubic',fill_value='extrapolate')

            fwhm = c_km_s/f(wave)
            expo = np.ones(len(fwhm))*2.0

            shape0 = spectrum.shape

            fwhm = fwhm.ravel()
            expo = expo.ravel()
            data_wave_temp = wave.ravel()
            spectrum = spectrum.ravel()

            ww = np.zeros_like(spectrum)
            sp2 = np.zeros_like(spectrum)

            range_scan =  np.max(fwhm)/(np.median(np.gradient(data_wave_temp)/data_wave_temp)*c_km_s)*3
            range_scan = int(np.ceil(range_scan))
            
            ew = (fwhm / 2) / (2 * np.log(2)) ** (1 / expo)
            
            for offset in range(-range_scan,range_scan):
                dv = (data_wave_temp/np.roll(data_wave_temp,offset)-1)*c_km_s
                w = supergauss(dv, ew, expo)
                sp2+=np.roll(spectrum,offset)*w
                ww+=w

            sp2/=ww
            
            telluric_spectrum_conv = sp2.reshape(shape0)


            #Antoine
            # f=interp1d(data_wave,Resolution_map[Order],kind='cubic',fill_value='extrapolate')
            # def fct_ker(w, w_center):
            #     res = f(w_center)                
            #     fwhm = w_center/res
            #     sigma = fast_conv.fwhm2sigma(fwhm)

            #     return fast_conv.gaussians(w, w_center, sigma)

            # idx_pad = int(np.ceil(np.max(c_km_s/f(wave))/(np.median(np.gradient(wave)/wave)*c_km_s)*3))  # Maximum shift in grid index for the kernel
            
            # data_wave_temp = wave[idx_pad:-idx_pad]

            

            # idx_grid = np.arange(len(data_wave_temp), dtype=int) + idx_pad
            # offset_idx = np.arange(-idx_pad, idx_pad+1, dtype=int)
            # idx = idx_grid[None, :] + offset_idx[:, None]
            # grid_compact_kernel = wave[idx]
            
            # compact_matrix_alt = fct_ker(grid_compact_kernel,data_wave_temp)
            
            # # Make it sparse
            # conv_matrix_alt_2 = fast_conv.get_c_matrix(compact_matrix_alt, wave, bounds=(data_wave_temp[0],data_wave_temp[-1]))
            # telluric_spectrum_conv = conv_matrix_alt_2.dot(telluric_spectrum)
                
                

            # step_convolution = 150
            # data_wave_temp = wave
            # tell_temp= telluric_spectrum
            # telluric_spectrum_conv = 1. * np.ones(len(data_wave_temp))
            # for p in range(len(data_wave_temp))[step_convolution:-step_convolution]:
            #     if tell_temp[p]<0.999:
            #         if np.searchsorted(data_wave,data_wave_temp[p-step_convolution]) >= len(data_wave):
            #             index_pixel     = np.searchsorted(data_wave,data_wave_temp[p-step_convolution])-1
            #         else:
            #             index_pixel     = np.searchsorted(data_wave,data_wave_temp[p-step_convolution])
            #         resolution_pixel    = Resolution_map[Order,index_pixel]
            #         fwhm_angstrom_pixel = data_wave_temp[p]/resolution_pixel
            #         sigma_angstrom      = fwhm_angstrom_pixel / 2.3548
            #         wave_psf            = data_wave_temp[p-int(np.ceil(4*sigma_angstrom/step)):p+int(np.ceil(4*sigma_angstrom/step))+1]
            #         flux_psf            = tell_temp[p-int(np.ceil(4*sigma_angstrom/step)):p+int(np.ceil(4*sigma_angstrom/step))+1]
            #         gaussian_psf        = np.multiply( np.array(1. / (sigma_angstrom * np.sqrt( 2. * np.pi ) ) ) , np.exp( - np.true_divide( ( wave_psf - data_wave_temp[p] ) **2. , np.array( 2. * sigma_angstrom ** 2. ) ) ) )
            #         gaussian_psf_norm   = np.true_divide(gaussian_psf,np.array(np.sum(gaussian_psf)))
            #         telluric_spectrum_conv[p] = np.convolve(flux_psf,gaussian_psf_norm,mode='same')[int(len(wave_psf)/2)]
            #     else:
            #         telluric_spectrum_conv[p] = tell_temp[p]


        telluric_spectrum_interp=np.empty(len(data_wave))
        Dl = np.concatenate(([np.diff(data_wave)[0]],np.diff(data_wave)))/2.
        for i in range(len(data_wave)):
            telluric_spectrum_interp[i] = np.mean(telluric_spectrum_conv[np.searchsorted(data_wave_temp,data_wave[i]-Dl[i]):np.searchsorted(data_wave_temp,data_wave[i]+Dl[i])])#*step


        ###########################
        ####    Compute CCFs   ####
        ###########################

        wave_line_ccf = lines_position_fit['CCF_lines_position_wavelength'][ (data_wave[0]+4.*born < lines_position_fit['CCF_lines_position_wavelength']) &  (lines_position_fit['CCF_lines_position_wavelength'] < data_wave[-1]-4.*born)]
        sij_ccf       = np.ones(len(wave_line_ccf))


        ccf_uncorr       = compute_CCF(data_wave[np.isnan(data_flux)==False],data_flux[np.isnan(data_flux)==False],                          wave_line_ccf,sij_ccf,rv,np.diff(rv)[0],0,0,0,instrument='ESPRESSO',rescale=0)
        ccf_model_conv   = compute_CCF(data_wave[np.isnan(data_flux)==False],telluric_spectrum_interp[np.isnan(data_flux)==False],           wave_line_ccf,sij_ccf,rv,np.diff(rv)[0],0,0,0,instrument='ESPRESSO',rescale=0)

        ######################
        ####    Outputs   ####
        ######################
        ccf_uncorr_ord.append(ccf_uncorr/np.nanmedian(ccf_uncorr[np.abs(rv)>15]))
        ccf_model_conv_ord.append(ccf_model_conv/np.nanmedian(ccf_model_conv[np.abs(rv)>15]))

    ###########################################
    ####    Combine CCFs and minimization  ####
    ###########################################

    ccf_uncorr_ord          = np.array(ccf_uncorr_ord)
    ccf_model_conv_ord      = np.array(ccf_model_conv_ord)

    ccf_uncorr_master       = np.mean(ccf_uncorr_ord,axis=0)
    ccf_model_conv_master   = np.mean(ccf_model_conv_ord,axis=0)


    return (ccf_uncorr_master / ccf_model_conv_master-1)



def compute_telluric_model(Fitted_Parameters,Molecules,M_mol_molecules,N_x_molecules,database,qt_list,data_wavelength,Orders,Resolution_map,instrument):
    """ Function to apply the best telluric absorption model over the all spectral range"""
    ### Comment: this function is really similar to "result_tellurique_layers_update" but it is simplified as there is no need to select only the 10 strongest lines
    
    
    timer_start =time.time()
    time_spectrum=0
    time_conv=0
    time_co_m=0
    time_bin=0
    
	##################################
	####    Constants and Inputs  ####
	##################################
    
    
    R     = 3000000 # model resolution

    c     = 299792458.0                 # [m*s^-1]
    c_km_s = c/1000
    NA    = 6.02214086e+23              # [mol^-1]
    k_b   = 1.3806503e-23               # [J*K^-1] = [kg*m^2*s^-2]
    c2    = 1.43880285                  # c*h/k in [cm*K]
    
    
    telluric_spectrum_interp_ord    = []
        
    for Ord in range(len(Orders)):
        
        
        Order=Orders[Ord]
        
        data_wave = data_wavelength[Order]
        step      = data_wave[len(data_wave)//2]/R
        born      = data_wave[len(data_wave)//2]*40/300000

        
        
        wave                     = np.arange(data_wave[0]-born,data_wave[-1]+born,step) #create synthetic spectrum on a broader range
        wave_wavenumber          = 1./(wave*10**-8)
        wave_wavenumber_selected = wave_wavenumber
        
        
        spectrum=np.zeros(len(wave))
        
        for Mol in range(len(Molecules)):
            

            temp        = Fitted_Parameters[Mol].params['Temperature'].value
            P_0         = Fitted_Parameters[Mol].params['Pressure_ground'].value
            PWV_airmass = Fitted_Parameters[Mol].params['PWV_w_airmass'].value
        
            
            hitran_database = database[Mol]
            qt_used         = qt_list[Mol]['Qt'][np.where(qt_list[Mol]['Temperature']==round(temp))[0]][0]
            M_mol           = M_mol_molecules[Mol]
            N_x             = N_x_molecules[Mol]
            
            wave_number_hitran_rest = hitran_database['wave_number'][(wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
            intensity_hitran        = hitran_database['Intensity'][(wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
            gamma_air               = hitran_database['gamma_air'][(wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
            n_air                   = hitran_database['N'][(wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
            delta_air               = hitran_database['delta'][(wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
            Epp                     = hitran_database['Epp'][(wave_wavenumber_selected[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wave_wavenumber_selected[0])]
            
            
            ########################################################
            ####    Compute basic parameters for telluric model ####
            ########################################################
            
            intensity_hitran          = intensity_hitran * ( 174.5813 / qt_used ) * ( np.exp( - c2 * Epp / temp ) ) / ( np.exp( - c2 * Epp / 296.0 ) ) * ( 1.- np.exp( - c2 * wave_number_hitran_rest / temp ) ) / ( 1.- np.exp( - c2 * wave_number_hitran_rest / 296.0 ) )
            wave_number_hitran_P0     = wave_number_hitran_rest + delta_air * P_0
            hwhm_lorentzian_hitran_P0 = ( 296.0 / temp ) ** n_air *  gamma_air *  P_0     #assuming P_mol=0       
            if Molecules[Mol] != 'H2O':
                hwhm_gaussian_hitran  = wave_number_hitran_rest / c * np.sqrt( 2. * NA * k_b * temp * np.log(2) / ( 10**-3 * M_mol) )

            
            ####################################
            ####    Compute telluric model  ####
            ####################################

            if len(wave_number_hitran_rest)!=0:
                
                start_spectrum = time.time()
                for i in range(len(wave_number_hitran_P0)):
                
                    wave_wavenumber_i = wave_wavenumber[np.searchsorted(-wave_wavenumber,-10**8/(10**8/(wave_number_hitran_P0[i])-8.*born)):np.searchsorted(-wave_wavenumber,-10**8/(10**8/(wave_number_hitran_P0[i])+8.*born))]
 
                    if Molecules[Mol] == 'H2O': #lorentzian profile
                        line_profile = 1. / np.pi * hwhm_lorentzian_hitran_P0[i] / ( hwhm_lorentzian_hitran_P0[i]**2. + ( wave_wavenumber_i - wave_number_hitran_P0[i] )**2. )
                    else:
                        line_profile = voigt(wave_wavenumber_i, HWHM=hwhm_gaussian_hitran[i], gamma=hwhm_lorentzian_hitran_P0[i], center=wave_number_hitran_P0[i])
                    
                    Profile_n=np.zeros(len(wave_wavenumber))
                    Profile_n[np.searchsorted(-wave_wavenumber,-wave_wavenumber_i)]=line_profile # this is done to adapt the x-scale to the full spectral range
                    
                    spectrum = np.add(spectrum, PWV_airmass * N_x * intensity_hitran[i] * Profile_n)
                
                end_spectrum = time.time()
                time_spectrum = np.add(time_spectrum,end_spectrum-start_spectrum)

                    
        telluric_spectrum = np.exp( - spectrum )
        

            
        ######################################
        ####    Convolution and binning   ####
        ######################################
        start_conv = time.time()
        if np.min(telluric_spectrum)==1:
            telluric_spectrum_interp=np.ones(len(data_wave))  
        else:
            if instrument == 'SPIRou_as_NIRPS': #to be removed for Danuta
                data_wave_temp = wave
                beta=2.24 #top sqaure for spirou
                resolution_pixel    = 60000
                fwhm_angstrom_pixel = wave[len(wave)//2]/resolution_pixel
                sigma_angstrom      = fwhm_angstrom_pixel / 2.3548  
                wave_psf            = wave[len(wave)//2-int(np.ceil(4*sigma_angstrom/step)):len(wave)//2+int(np.ceil(4*sigma_angstrom/step))+1]
                gaussian_psf        = np.multiply( np.array(1. / (sigma_angstrom * np.sqrt( 2. * np.pi ) ) ) , np.exp( - np.true_divide( np.abs( wave_psf - wave[len(wave)//2] ) **beta , np.array( 2. * sigma_angstrom ** beta ) ) ) )
                gaussian_psf_norm   = np.true_divide(gaussian_psf,np.array(np.sum(gaussian_psf)))
                telluric_spectrum_conv = np.convolve(telluric_spectrum,gaussian_psf_norm,mode='same')                        
            else:
                
                start_co_m = time.time()


                spectrum = telluric_spectrum#np.ones(len(telluric_spectrum)) #telluric_spectrum
                f=interp1d(data_wave,Resolution_map[Order],kind='cubic',fill_value='extrapolate')

                fwhm = c_km_s/f(wave)
                expo = np.ones(len(fwhm))*2.0

                shape0 = spectrum.shape

                fwhm = fwhm.ravel()
                expo = expo.ravel()
                data_wave_temp = wave.ravel()
                spectrum = spectrum.ravel()

                ww = np.zeros_like(spectrum)
                sp2 = np.zeros_like(spectrum)

                range_scan =  np.max(fwhm)/(np.median(np.gradient(data_wave_temp)/data_wave_temp)*c_km_s)*3
                range_scan = int(np.ceil(range_scan))
                
                ew = (fwhm / 2) / (2 * np.log(2)) ** (1 / expo)

                for offset in range(-range_scan,range_scan):
                    dv = (data_wave_temp/np.roll(data_wave_temp,offset)-1)*c_km_s
                    w = supergauss(dv, ew, expo)
                    sp2+=np.roll(spectrum,offset)*w
                    ww+=w

                sp2/=ww
                
                # fwhm = fwhm.reshape(shape0)
                # expo = expo.reshape(shape0)
                # data_wave_temp = data_wave_temp.reshape(shape0)
                # spectrum = spectrum.reshape(shape0)
                telluric_spectrum_conv = sp2.reshape(shape0)
                end_co_m = time.time()

                
                # # def gaussianfit(x,c,a,x0,sigma):
                # #     "function of a gaussian"
                # #     #a=1./(sigma*np.sqrt(2*np.pi))
                # #     return c+a*np.exp(-0.5*np.abs((x-x0)/(sigma))**2)
                # # from lmfit import Model
                # # gmodel = Model(gaussianfit)        
                # # params = Parameters()
                # # params.add('c',   value= 1, vary=True  )
                # # params.add('a',   value= -0.1, vary=True  )
                # # params.add('x0',   value= wave[ind_test], vary=True  )
                # # params.add('sigma',   value= fwhm[ind_test]/(2*np.sqrt(2*np.log(2))),  min=0., vary=True  )

                # # result_E = gmodel.fit(sp2[ind_test-1000:ind_test+1000], params, x=wave[ind_test-1000:ind_test+1000], )
                # # print(result_E.fit_report())

                


                # start_co_m = time.time()


                # #Antoine
                # f=interp1d(data_wave,Resolution_map[Order],kind='cubic',fill_value='extrapolate')
                # def fct_ker(w, w_center):
                #     res = f(w_center)                
                #     fwhm = w_center/res
                #     sigma = fast_conv.fwhm2sigma(fwhm)
    
                #     return fast_conv.gaussians(w, w_center, sigma)

                # idx_pad = int(np.ceil(np.max(c_km_s/f(wave))/(np.median(np.gradient(wave)/wave)*c_km_s)*3))  # Maximum shift in grid index for the kernel
                
                # data_wave_temp = wave[idx_pad:-idx_pad]

                

                # idx_grid = np.arange(len(data_wave_temp), dtype=int) + idx_pad
                # offset_idx = np.arange(-idx_pad, idx_pad+1, dtype=int)
                # idx = idx_grid[None, :] + offset_idx[:, None]
                # grid_compact_kernel = wave[idx]
                
                # compact_matrix_alt = fct_ker(grid_compact_kernel,data_wave_temp)
                
                # # Make it sparse
                # conv_matrix_alt_2 = fast_conv.get_c_matrix(compact_matrix_alt, wave, bounds=(data_wave_temp[0],data_wave_temp[-1]))
                # telluric_spectrum_conv = conv_matrix_alt_2.dot(telluric_spectrum)
                # end_co_m = time.time()
                
                
                # result_A = gmodel.fit(telluric_spectrum_conv_test[ind_test-1000:ind_test+1000], params, x=data_wave_temp[ind_test-1000:ind_test+1000], )
                
                # start_co_m = time.time()
                # convolution_matrix = fast_conv.get_c_matrix(fct_ker, wave, bounds=(data_wave_temp[0],data_wave_temp[-1]), thresh=1e-6)
                # end_co_m = time.time()

                # telluric_spectrum_conv = convolution_matrix.dot(telluric_spectrum)
                
                time_co_m = np.add(time_co_m,end_co_m-start_co_m)
               
                # step_convolution = 150
                # data_wave_temp = wave
                # tell_temp= telluric_spectrum
                # telluric_spectrum_conv_org = 1. * np.ones(len(data_wave_temp))
                # sigs=[]
                # for p in range(len(data_wave_temp))[step_convolution:-step_convolution]:
                #     if tell_temp[p]<0.999:
                #         if np.searchsorted(data_wave,data_wave_temp[p-step_convolution]) >= len(data_wave):
                #             index_pixel     = np.searchsorted(data_wave,data_wave_temp[p-step_convolution])-1
                #         else:
                #             index_pixel     = np.searchsorted(data_wave,data_wave_temp[p-step_convolution])
                #         resolution_pixel    = Resolution_map[Order,index_pixel]
                #         if np.isnan(resolution_pixel):
                #             resolution_pixel    = np.nanmin(Resolution_map[Order])
        
                #         fwhm_angstrom_pixel = data_wave_temp[p]/resolution_pixel
                #         sigma_angstrom      = fwhm_angstrom_pixel / 2.3548
                #         wave_psf            = data_wave_temp[p-int(np.ceil(4*sigma_angstrom/step)):p+int(np.ceil(4*sigma_angstrom/step))+1]
                #         sigs.append(int(np.ceil(4*sigma_angstrom/step)))
                #         flux_psf            = tell_temp[p-int(np.ceil(4*sigma_angstrom/step)):p+int(np.ceil(4*sigma_angstrom/step))+1]
                #         gaussian_psf        = np.multiply( np.array(1. / (sigma_angstrom * np.sqrt( 2. * np.pi ) ) ) , np.exp( - np.true_divide( ( wave_psf - data_wave_temp[p] ) **2. , np.array( 2. * sigma_angstrom ** 2. ) ) ) )
                #         gaussian_psf_norm   = np.true_divide(gaussian_psf,np.array(np.sum(gaussian_psf)))
                #         telluric_spectrum_conv_org[p] = np.convolve(flux_psf,gaussian_psf_norm,mode='same')[int(len(wave_psf)/2)]
                #     else:
                #         telluric_spectrum_conv_org[p] = tell_temp[p]
                
            end_conv = time.time()

            start_bin = time.time()

                    
            telluric_spectrum_interp=np.empty(len(data_wave))
            Dl = np.concatenate(([np.diff(data_wave)[0]],np.diff(data_wave)))/2.
            for i in range(len(data_wave)):
                telluric_spectrum_interp[i] = np.mean(telluric_spectrum_conv[np.searchsorted(data_wave_temp,data_wave[i]-Dl[i]):np.searchsorted(data_wave_temp,data_wave[i]+Dl[i])])#*step

            end_bin = time.time()

        time_conv = np.add(time_conv,end_conv-start_conv)
        time_bin  = np.add(time_bin,end_bin-start_bin)
        
        
            
        telluric_spectrum_interp_ord.append(telluric_spectrum_interp)
        

        
    telluric_spectrum_interp_ord    = np.array(telluric_spectrum_interp_ord)
    
    timer_end =time.time()
    logger.info('-------------------')
    logger.info('Spectrum time            : %s',np.round(time_spectrum/60.,2))
    logger.info('Convolution time         : %s',np.round(time_conv/60.,2))
    logger.info('Conv matrix time         : %s',np.round(time_co_m/60.,2))
    logger.info('Binning time             : %s',np.round(time_bin/60.,2))
    logger.info('Full range spectrum time : %s',np.round((timer_end-timer_start)/60.,2))
    return telluric_spectrum_interp_ord

def open_resolution_map(instrument,time_science,ins_mode,bin_x=2):
    """Open static resolution map. It dependens on the instrumental mode of the science frame and on the epoch where it was acquired ==> technical intervention"""
    if instrument =='ESPRESSO':
        if ins_mode == 'SINGLEUHR':
            if time_science < 2458421.5:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2018-09-09T12:18:49.369_RESOLUTION_MAP.fits')
                period = 'UHR_1x1_pre_october_2018'
            elif (time_science > 2458421.5) and (time_science < 2458653.1):
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2019-04-06T11:52:27.477_RESOLUTION_MAP.fits')
                period = 'UHR_1x1_post_october_2018'
            elif time_science > 2458653.1:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2019-11-06T11:06:36.913_RESOLUTION_MAP.fits')
                period = 'UHR_1x1_post_june_2019'
            
        elif (ins_mode == 'MULTIMR') and (bin_x == 4):
            if time_science < 2458421.5:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2018-07-08T14:51:53.873_RESOLUTION_MAP.fits')
                period = 'MR_4x2_pre_october_2018'
            elif (time_science > 2458421.5) and (time_science < 2458653.1):
                instrumental_function = fits.open('Static_resolution/r.ESPRE.2018-12-01T11:25:27.377_RESOLUTION_MAP.fits')
                period = 'MR_4x2_post_october_2018'
            elif time_science > 2458653.1:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2019-11-05T12:23:45.139_RESOLUTION_MAP.fits')
                period = 'MR_4x2_post_june_2019'
        
        elif (ins_mode == 'MULTIMR') and (bin_x == 8):
            if time_science < 2458421.5:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2018-07-06T11:48:22.862_RESOLUTION_MAP.fits')
                period = 'MR_8x4_pre_october_2018'
            elif (time_science > 2458421.5) and (time_science < 2458653.1):
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2018-10-24T14:16:52.394_RESOLUTION_MAP.fits')
                period = 'MR_8x4_post_october_2018'
            elif time_science > 2458653.1:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2019-11-05T12:59:57.138_RESOLUTION_MAP.fits')
                period = 'MR_8x4_post_june_2019'
        
        elif (ins_mode == 'SINGLEHR') and (bin_x == 1):
            if time_science < 2458421.5:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2018-07-04T21:28:53.759_RESOLUTION_MAP.fits')
                period = 'HR_1x1_pre_october_2018'
            elif (time_science > 2458421.5) and (time_science < 2458653.1):
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2019-01-05T19:53:55.501_RESOLUTION_MAP.fits')
                period = 'HR_1x1_post_october_2018'
            elif time_science > 2458653.1:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2019-11-19T10:10:12.384_RESOLUTION_MAP.fits')
                period = 'HR_1x1_post_june_2019'
        
        elif (ins_mode == 'SINGLEHR') and (bin_x == 2):
            if time_science < 2458421.5:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2018-09-05T14:01:58.063_RESOLUTION_MAP.fits')
                period = 'HR_2x1_pre_october_2018'
            elif (time_science > 2458421.5) and (time_science < 2458653.1):
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2019-03-30T21:28:32.060_RESOLUTION_MAP.fits')
                period = 'HR_2x1_post_october_2018'
            elif time_science > 2458653.1:
                instrumental_function = fits.open('Static_resolution/ESPRESSO/r.ESPRE.2019-09-26T11:06:01.271_RESOLUTION_MAP.fits')
                period = 'HR_2x1_post_june_2019'

    if (instrument =='NIRPS_drs') or (instrument =='NIRPS_apero'):
        if ins_mode == 'HA':
            if time_science < 2459850.5:
                instrumental_function = fits.open('Static_resolution/NIRPS/r.NIRPS.2022-06-15T18:09:53.175_RESOLUTION_MAP.fits')
                period = 'HA_COMM5'
            elif (time_science > 2459850.5):
                instrumental_function = fits.open('Static_resolution/NIRPS/r.NIRPS.2022-11-28T21:06:04.212_RESOLUTION_MAP.fits')
                period = 'HA_COMM7'
        elif ins_mode == 'HE':
            if time_science < 2459850.5:
                instrumental_function = fits.open('Static_resolution/NIRPS/r.NIRPS.2022-06-15T17:52:36.533_RESOLUTION_MAP.fits')
                period = 'HE_COMM5'
            elif (time_science > 2459850.5):
                instrumental_function = fits.open('Static_resolution/NIRPS/r.NIRPS.2022-11-28T20:47:51.815_RESOLUTION_MAP.fits')
                period = 'HE_COMM7'        

        logger.info(period)
    resolution_map=instrumental_function[1].data
    instrumental_function.close()
    return resolution_map

def save_file(path,instrument,flux_corr,flux_err_corr,telluric,result_fit,molecules,save_type,save_location):


    if save_type == 'DRS':
        os.system('cp '+path+' '+save_location+path.split('/')[-1][:-7]+'_TELL_CORR_A.fits')
        file = save_location+path.split('/')[-1][:-7]+'_TELL_CORR_A.fits'        
        hdr = fits.Header()
        hdr['EXTNAME']='SCIDATA'
        fits.update( file, np.where(telluric>0.1,flux_corr,0),1,header=hdr)

        header_files = fits.open(file)
        # Modify HIERARCH ESO PRO CATG
        header_files[0].header['HIERARCH ESO PRO CATG'] = 'S2D_BLAZE_TELL_CORR_A'

        
    elif save_type == 'Extended':
        os.system('cp '+path+' '+save_location+path.split('/')[-1][:-7]+'_TELL_CORR_EXTENDED_A.fits')
        file = save_location+path.split('/')[-1][:-7]+'_TELL_CORR_EXTENDED_A.fits'        
        hdr = fits.Header()
        hdr['EXTNAME']='SCIDATA_CORR'
        fits.append( file, np.where(telluric>0.1,flux_corr,0) ,header=hdr)
        hdr['EXTNAME']='ERRDATA_CORR'
        fits.append( file, flux_err_corr,header=hdr)
        hdr['EXTNAME']='TELLURIC'
        fits.append( file, telluric,header=hdr) 

        header_files = fits.open(file)
        # Modify HIERARCH ESO PRO CATG
        header_files[0].header['HIERARCH ESO PRO CATG'] = 'S2D_BLAZE_TELL_CORR_EXTENDED_A'
        
    elif save_type == 'Telluric':
        os.system('cp '+path+' '+save_location+path.split('/')[-1][:-13]+'_TELL_A.fits')
        file = fits.open(save_location+path.split('/')[-1][:-13]+'_TELL_A.fits')
        file.pop('SCIDATA')
        file.pop('ERRDATA')
        file.writeto(save_location+path.split('/')[-1][:-13]+'_TELL_A.fits',overwrite=True)
        file.close()
        
        file = save_location+path.split('/')[-1][:-13]+'_TELL_A.fits'
        hdr = fits.Header()
        hdr['EXTNAME']='TELLURIC'
        fits.append( file, telluric,header=hdr) 

        header_files = fits.open(file)
        # Modify HIERARCH ESO PRO CATG
        header_files[0].header['HIERARCH ESO PRO CATG'] = 'S2D_TELL_A'

    
    for M in range(len(molecules)):
        header_files[0].header['ESO QC TELL '+ molecules[M] +' IWV']          = result_fit[M].params['PWV_w_airmass'].value*10 #to put in mm

        try:
            header_files[0].header['ESO QC TELL '+ molecules[M] +' IWV ERR']      = result_fit[M].params['PWV_w_airmass'].stderr*10 #to put in mm
        except TypeError:
            header_files[0].header['ESO QC TELL '+ molecules[M] +' IWV ERR']      = 9999
        except ValueError:
            header_files[0].header['ESO QC TELL '+ molecules[M] +' IWV ERR']      = 9999
        header_files[0].header['ESO QC TELL '+ molecules[M] +' PRESSURE']     = result_fit[M].params['Pressure_ground'].value*1013.2501   
 
        try:
            header_files[0].header['ESO QC TELL '+ molecules[M] +' PRESSURE ERR'] = result_fit[M].params['Pressure_ground'].stderr*1013.2501
        except TypeError:
            header_files[0].header['ESO QC TELL '+ molecules[M] +' PRESSURE ERR']      = 9999
        except ValueError:
            header_files[0].header['ESO QC TELL '+ molecules[M] +' PRESSURE ERR']      = 9999



        header_files[0].header['ESO QC TELL '+ molecules[M] +' TEMP']         = result_fit[M].params['Temperature'].value - 273.15
        if molecules != 'H2O':
            header_files[0].header['ESO QC TELL '+ molecules[M] +' TEMP ERR'] = result_fit[M].params['Temperature'].stderr       
        header_files[0].header['ESO QC TELL '+ molecules[M] +' CHI SQUARE']   = result_fit[M].chisqr

    header_files.writeto(file,overwrite=True)
    header_files.close()
    return

def Run_ATC(Input,options):
    
    logger.info(Input)

    
    instrument=options[0]
    molecules=options[1]
    
    
    ###################
    ####  HITRAN   ####
    ###################
    hitran_database_full_range_molecules  = np.empty(len(molecules),dtype=object)
    hitran_database_lines_model_molecules = np.empty(len(molecules),dtype=object)
    hitran_database_lines_fit_molecules   = np.empty(len(molecules),dtype=object)
    qt_file_molecules                     = np.empty(len(molecules),dtype=object)
    N_x_molecules                         = np.empty(len(molecules))
    M_mol_molecules                       = np.empty(len(molecules))
    temp_bool_molecules                   = np.empty(len(molecules),dtype=bool)
    temp_offset_molecules                 = np.empty(len(molecules))
    pwv_value_molecules                   = np.empty(len(molecules))
    
    for m in range(len(molecules)):
        
        if instrument.count('NIRPS')==1:
            static_file_full_range = fits.open('Static_model/'+'NIRPS'+'/Static_hitran_qt_'+molecules[m]+'.fits')
            static_file_lines_fit  = fits.open('Static_model/'+'NIRPS'+'/Static_hitran_strongest_lines_'+molecules[m]+'.fits')
        else:
            static_file_full_range = fits.open('Static_model/'+instrument+'/Static_hitran_qt_'+molecules[m]+'.fits')
            static_file_lines_fit  = fits.open('Static_model/'+instrument+'/Static_hitran_strongest_lines_'+molecules[m]+'.fits')

        hitran_database_full_range_molecules[m]  = static_file_full_range[1].data
        qt_file_molecules[m]                     = static_file_full_range[2].data
        
        hitran_database_lines_model_molecules[m] = static_file_lines_fit[1].data        
        hitran_database_lines_fit_molecules[m]   = static_file_lines_fit[2].data 
        
        static_file_full_range.close()
        static_file_lines_fit.close()
        if molecules[m]=='H2O':
            N_x_molecules[m]   = 3.3427274952610645e+22      # [molecules*cm^-3]
            M_mol_molecules[m] = 18.01528                    # [g*mol^-1]
            temp_bool_molecules[m]  = False
            temp_offset_molecules[m] = 0
        if molecules[m]=='CH4':
            N_x_molecules[m]   = 4.573e+13                   # [molecules*cm^-3]
            M_mol_molecules[m] = 16.04                       # [g*mol^-1]
            temp_bool_molecules[m]  = False
            temp_offset_molecules[m] = 20
            pwv_value_molecules[m]  = 2163379#1171000        # [cm]
        if molecules[m]=='CO2':
            N_x_molecules[m]   = 9.8185e+15                  # [molecules*cm^-3]
            M_mol_molecules[m] = 44.01                       # [g*mol^-1]
            temp_bool_molecules[m]  = False
            temp_offset_molecules[m] = 20
            pwv_value_molecules[m]  = 951782#1171000         # [cm]
        if molecules[m]=='O2':
            N_x_molecules[m]   = 5.649e+18                   # [molecules*cm^-3]
            M_mol_molecules[m] = 31.9988                     # [g*mol^-1]
            temp_bool_molecules[m]  = False
            temp_offset_molecules[m] = 20
            pwv_value_molecules[m]  = 660128                 # [cm]

    ###################
    ####    DATA   ####
    ###################
    data=fits.open(Input)
    
    ####################
    ####################
    ####   Inputs   ####
    ####################
    ####################
    if instrument=='ESPRESSO':
        #Telescope parameters
        data_header=data[0].header
        UT_used=data_header['TELESCOP'][-1]
        airmass   = ( data_header['HIERARCH ESO TEL'+UT_used+' AIRM START'] + data_header['HIERARCH ESO TEL'+UT_used+' AIRM END'] ) / 2.
        PWV_AM1   = ( data_header['HIERARCH ESO TEL'+UT_used+' AMBI IWV START'] + data_header['HIERARCH ESO TEL'+UT_used+' AMBI IWV END'] ) / 2.
        temp      = data_header['HIERARCH ESO TEL'+UT_used+' AMBI TEMP'] + 273.15
        P_tot     = ( data_header['HIERARCH ESO TEL'+UT_used+' AMBI PRES START'] + data_header['HIERARCH ESO TEL'+UT_used+' AMBI PRES END'] ) / 2. /1013.2501
        berv      = data_header['HIERARCH ESO QC BERV']
        PWV_airmass = PWV_AM1  * airmass / 10.0 * 1.00 # /10. to put in cm
    
        #Science data
        data_wave = (data[4].data)/((1.+1.55e-8)*(1.+berv/299792.458))
        data_flux = data[1].data
        data_flux_err = data[2].data
        
        #Resolution map
        Resolution_instrumental_map = open_resolution_map(instrument,data_header['HIERARCH ESO QC BJD'],data_header['HIERARCH ESO INS MODE'],data_header['HIERARCH ESO DET BINX'])
        
        #CCF grid
        rvs= np.arange(-40,40.001,0.5)
    
    elif instrument=='SPIRou_as_NIRPS':
        data_header=data[0].header
        airmass   = data_header['AIRMASS']
        PWV_AM1   = 1 #[mm]
        temp      = data_header['TEA6FOOT'] + 273.15
        P_tot     = data_header['PRESSURE'] /1013.2501
        PWV_airmass = PWV_AM1  * airmass / 10.0 * 1.00 # /10. to put in cm
        
        #Science data
        data_wave = fits2wave(Input)*10
        data_flux = data[0].data
        data_flux_err = np.sqrt(abs(data_flux)+data_header['RDNOISE']**2)

        #Resolution map
        Resolution_instrumental_map = None#open_resolution_map(data_header['HIERARCH ESO QC BJD'],data_header['HIERARCH ESO INS MODE'],data_header['HIERARCH ESO DET BINX'])
        
        
        #CCF grid
        rvs= np.arange(-40,40.001,1.0)        

    elif instrument=='NIRPS_apero':
        data_header=data[0].header
        airmass   = ( data_header['HIERARCH ESO TEL AIRM START'] + data_header['HIERARCH ESO TEL AIRM END'] ) / 2.

        PWV_AM1   = 1 #[mm]
        temp      = data_header['HIERARCH ESO TEL AMBI TEMP'] + 273.15
        P_tot     = ( data_header['HIERARCH ESO TEL AMBI PRES START'] + data_header['HIERARCH ESO TEL AMBI PRES END'] ) / 2. /1013.2501
        PWV_airmass = PWV_AM1  * airmass / 10.0 * 1.00 # /10. to put in cm
        
        #Science data
        data_wave = fits2wave(Input)*10
        data_flux = data[1].data
        data_flux_err = np.sqrt(abs(data_flux)) #+data_header['RDNOISE']**2)

        #Resolution map
        Resolution_instrumental_map = open_resolution_map(instrument,data_header['HIERARCH ESO QC BJD'],data_header['HIERARCH ESO INS MODE'])
        
        
        #CCF grid
        rvs= np.arange(-40,40.001,1.0)

    elif instrument=='NIRPS_drs':
        data_header=data[0].header
        airmass   = ( data_header['HIERARCH ESO TEL AIRM START'] + data_header['HIERARCH ESO TEL AIRM END'] ) / 2.
        PWV_AM1   = 1 
        temp      = data_header['HIERARCH ESO TEL AMBI TEMP'] + 273.15
        P_tot     = ( data_header['HIERARCH ESO TEL AMBI PRES START'] + data_header['HIERARCH ESO TEL AMBI PRES END'] ) / 2. /1013.2501


        berv      = data_header['HIERARCH ESO QC BERV']
        PWV_airmass = PWV_AM1  * airmass / 10.0 * 1.00 # /10. to put in cm
        
        #Science data
        data_wave = (data[4].data)/((1.+1.55e-8)*(1.+berv/299792.458))
        data_flux = data[1].data
        data_flux_err = data[2].data

        #Resolution map
        Resolution_instrumental_map = open_resolution_map(instrument,data_header['HIERARCH ESO QC BJD'],data_header['HIERARCH ESO INS MODE'])
        
        #CCF grid
        rvs= np.arange(-40,40.001,1.0)  

    data.close()
    
    #fit by molecules
    result_fit_molecules=np.empty(len(molecules),dtype=object)
    for m in range(len(molecules)):
        #Fit parameters initialization
        timer_start=time.time()
        params = Parameters()
        if molecules[m]=='H2O':
            params.add('PWV_w_airmass',   value= PWV_airmass,  min=0., vary=True  )
            params.add('Pressure_ground', value= P_tot/2.,     min=0., max=P_tot,  vary=True  )
        else:
            params.add('PWV_w_airmass',   value= pwv_value_molecules[m]*airmass,  min=0., vary=True  )                
            params.add('Pressure_ground', value= P_tot/4.,     min=0., max=P_tot,  vary=True  )
        
        params.add('Temperature',     value= temp-temp_offset_molecules[m],         min=150., max=temp+0,    vary=temp_bool_molecules[m] )
        
        
        #Fit minimization
        result_fit = minimize(fit_telluric_model, params, args=(rvs,[data_wave,data_flux,hitran_database_lines_model_molecules[m],qt_file_molecules[m],hitran_database_lines_fit_molecules[m],Resolution_instrumental_map,N_x_molecules[m],M_mol_molecules[m],molecules[m],instrument]))
        result_fit_molecules[m] = result_fit
        logger.info('%s',fit_report(result_fit))
        
        timer_end=time.time()
        logger.info('')
        logger.info('-------------------')
        logger.info('Fitting time for %s : %s',molecules[m],np.round((timer_end-timer_start)/60.,2))
        logger.info('-------------------')
        logger.info('')
    
    #Apply telluric correction to the full spectrum
    telluric_spectrum = compute_telluric_model(result_fit_molecules,molecules,M_mol_molecules,N_x_molecules,hitran_database_full_range_molecules,qt_file_molecules,data_wave,np.arange(len(data_wave)),Resolution_instrumental_map,instrument)
    
    return Input,instrument,data_flux,data_flux_err,telluric_spectrum,result_fit_molecules,molecules


def Run_ATC_files(Input,options):

    logger.info('USING A TOTAL OF %s THREADS TO PERFORM TELLURIC CORRECTION', ne.nthreads)

    for i in range(len(Input)):
        time_per_file=time.time()
        output_ATC= Run_ATC(Input[i],options)
        for save_type in options[3]:
            save_file(output_ATC[0],output_ATC[1],output_ATC[2]/output_ATC[4],output_ATC[3]/output_ATC[4],output_ATC[4],output_ATC[5],output_ATC[6],save_type=save_type, save_location=options[2])
        logger.info('Total time             : %s',np.round((time.time()-time_per_file)/60.,2))

    return


def multiprocessing_ATC(Input,options,nthreads=1):
    chunks = np.array_split(Input,nthreads)
    # print(chunks)
    pool = Pool(processes=nthreads)
    
    fixed_param=partial(Run_ATC_files,options=options)
    # print(options)
    pool.map(fixed_param, chunks)

    return
