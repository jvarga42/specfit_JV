#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
#%%
from specfit_JV_lib import * 
import numpy as np
import emcee
import corner
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import multiprocessing as mp
import os
from scipy.optimize import nnls
from scipy import stats
import re

os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

################################################################
# setup model
################################################################

# run_name = '10_with_gap_DHS_3grainsizes_5-25um' #7_5-13_5um' #'DHS_3'
# run_name = '15_JVmix_with_forst_aerosol_5-25um' #5-25um'
# run_name = '11_DHS_3grainsizes_8-13um' 
# run_name = '08_dyn_nnls_DHS_3grainsizes_5-25um'
# run_name = '10_GRF_6grainsizes_5-25um'
# run_name = '16_DHS_3grainsizes_5-25um'
# run_name = '17_GRF_flaterr_6gs_5-25um'
# run_name = '18_DHS_flaterr_3gs_5-25um'
# run_name = '19_DHS_flaterr_3gs_MgFe_5-25um'
# run_name = '20_GRF_flaterr_6gs_MgFe_5-25um'
# run_name = '21_GRF_flaterr_6gs_MgFe_7-14um'
# run_name = '23_DHS_flaterr_3gs_MgFe_nofay_7-14um'
# run_name = '24_DHS_flaterr_3gs_with_annSiO2_5-25um'

run_names = [\
#'25_newDHS_3gs_5-25um', #finished
#'26_newDHS_no_ann_SiO2_3gs_5-25um', #finished
#'27_GRF_with_ann_SiO2_3gs_5-25um', #finished
#'28_GRF_no_ann_SiO2_3gs_5-25um', #finished
#'29_newDHS_5gs_5-25um', #finished
#'30_newDHS_no_ann_SiO2_5gs_5-25um', #finished 
#'31_GRF_with_ann_SiO2_5gs_5-25um', #finished
'32_GRF_no_ann_SiO2_5gs_5-25um', #finished
#'33_newDHS_4gs_5-25um', #finished
#'34_newDHS_no_ann_SiO2_4gs_5-25um', #finished
'35_GRF_with_ann_SiO2_4gs_5-25um', #finished
'36_GRF_no_ann_SiO2_4gs_5-25um', #finished
#'37_newDHS_2gs_5-25um', #finished
#'38_newDHS_no_ann_SiO2_2gs_5-25um', #finished
'39_GRF_with_ann_SiO2_2gs_5-25um', #finished
'40_GRF_no_ann_SiO2_2gs_5-25um', #finished
#'41_newDHS_3vgs_5-25um',
#'42_newDHS_no_ann_SiO2_3vgs_5-25um',
'43_GRF_with_ann_SiO2_3vgs_5-25um',
'44_GRF_no_ann_SiO2_3vgs_5-25um',
]
[
'45_oldDHS_5gs_5-25um',
'46_oldDHS_no_ann_SiO2_5gs_5-25um',
'47_oldDHS_4gs_5-25um',
'48_oldDHS_no_ann_SiO2_4gs_5-25um',
'49_oldDHS_3gs_5-25um',
'50_oldDHS_no_ann_SiO2_3gs_5-25um',
'51_oldDHS_3vgs_5-25um',
'52_oldDHS_no_ann_SiO2_3vgs_5-25um',
'53_oldDHS_2gs_5-25um',
'54_oldDHS_no_ann_SiO2_2gs_5-25um',
]
#'55_newDHS_amorph_only_3gs_5-25um',
#'56_GRF_amorph_only_3gs_5-25um',

'''
run_names = [\
#'25_newDHS_3gs_5-25um', #finished
'26_newDHS_no_ann_SiO2_3gs_5-25um',
'27_GRF_with_ann_SiO2_3gs_5-25um', 
'28_GRF_no_ann_SiO2_3gs_5-25um', 
'49_oldDHS_3gs_5-25um',
'50_oldDHS_no_ann_SiO2_3gs_5-25um',
]
'''

# three things to explore:
# do we need large (5 um sized) grains? 
# do we need annealed silica?
# which is the best set of opcities? newDHS, oldDHS, GRF?

#more fits: gs: [0.1,1.0,2.0]   [0.1,2.0]
# old set of opacities
wl_limits =  [5.0,25.0] #[7.0,14.0] # [14.0,25.0] #[8.0,13.0] # #[7.5,13.5] #um

# select opacities
###################

#!!!!! check phyllosilicates (saponite, Morris 2013) and diopside !!!!
# features between 5 and 8 um
# DF Tau, IRS 46 (absorption), RW Aur
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

for run_name in run_names:
    print(run_name)
        
    if '5gs' in run_name:
        gsize_list = [0.1,1.0,2.0,3.0,4.0,5.0]   
    elif '4gs' in run_name:
        gsize_list = [0.1,1.0,2.0,5.0]    
    elif '3gs' in run_name:
        gsize_list = [0.1,2.0,5.0]   
    elif '3vgs' in run_name:
        gsize_list = [0.1,1.0,2.0] 
    elif '2gs' in run_name:
        gsize_list = [0.1,2.0]  
        
    #opac_dir_GRF = '/Users/jvarga/Dokumentumok/MATISSE/pro/opacities/GRF_opacity/'
    opac_dir_GRF = '/allegro6/matisse/varga/pro/opacities/GRF_opacity/'

    opac_species_list_GRF = [\
    'MgOlivine0.1.Combined.Kappa',
    ##'Olivine0.1.Combined.Kappa',
    'Forsterite0.1.Combined.Kappa',
    'MgPyroxene0.1.Combined.Kappa',
    ##'Pyroxene0.1.Combined.Kappa',
    'Enstatite0.1.Combined.Kappa',
    ##'Fayalite0.1.Combined.Kappa',
    'Silica0.1.Combined.Kappa',
    ##'kappa_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat'
    ]
    if not('no_ann_SiO2' in run_name):
        #pass
        opac_species_list_GRF.append('kappa_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat')

    opac_fname_list_GRF=[]
    for species in opac_species_list_GRF:
        for gsize in gsize_list:
            opac_fname_list_GRF.append(species.replace('0.1','%.1f'%(gsize)))
            
    #opac_dir_DHS = '/Users/jvarga/Dokumentumok/MATISSE/pro/specfit_v1.2/QVAL/'
    opac_dir_DHS = '/allegro6/matisse/varga/pro/specfit_v1.2/QVAL/'

    if 'newDHS' in run_name:
        opac_species_list_DHS = [\
        'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
        'Q_Fo_Zeidler_DHS_f0.99_rv0.1.dat',
        'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
        'Q_Ens_Zeidler_DHS_f0.99_rv0.1.dat',
        'Q_Am_Silica_Kit_DHS_f0.70_rv0.1.dat',
        #'Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat',
        ]
        if not('no_ann_SiO2' in run_name):
            #pass
            opac_species_list_DHS.append('Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat')    
    elif 'oldDHS' in run_name:   
        opac_species_list_DHS = [\
        'Q_Am_Mgol_Jae_DHS_f0.70_rv0.1.dat',
        'Q_Fo_Suto_DHS_f0.99_rv0.1.dat',
        'Q_Am_Mgpyr_Dor_DHS_f0.70_rv0.1.dat',
        'Q_Ens_Jaeger_DHS_f0.99_rv0.1.dat',
        'Q_Am_Silica_Kit_DHS_f0.70_rv0.1.dat',
        #'Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat',
        ]
        if not('no_ann_SiO2' in run_name):
            #pass
            opac_species_list_DHS.append('Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat')    
      
    if 'DHS' in run_name:         
        opac_fname_list_DHS=[]
        for species in opac_species_list_DHS:
            for gsize in gsize_list:
                opac_fname_list_DHS.append(species.replace('0.1','%.1f'%(gsize)))
            
    opac_dir_JVmix = opac_dir_DHS
    opac_fname_list_JVmix=[
    'Q_Am_Mgolivine_Jae_DHS_f0.7_rv0.1.dat',
    'Q_Am_Mgolivine_Jae_DHS_f0.7_rv2.0.dat',
    'Q_Am_Mgolivine_Jae_DHS_f0.7_rv5.0.dat',
    'Q_Am_Mgpyroxene_Dor_DHS_f0.7_rv0.1.dat',
    'Q_Am_Mgpyroxene_Dor_DHS_f0.7_rv2.0.dat',
    'Q_Am_Mgpyroxene_Dor_DHS_f0.7_rv5.0.dat',
    'Q_Fors_aerosol.dat',
    'Q_Fo_Zeidler_DHS_f0.8_rv2.0.dat',
    'Q_Fo_Zeidler_DHS_f0.8_rv5.0.dat',
    'Q_En_Jaeger_DHS_f1.0_rv0.1.dat',
    'Q_En_Jaeger_DHS_f1.0_rv2.0.dat',
    'Q_En_Jaeger_DHS_f1.0_rv5.0.dat',
    'Q_Silica0.1.GRF.dat',
    'Q_Silica2.0.GRF.dat',
    'Q_Silica5.0.GRF.dat',
    ]

    # fitter options
    ###################

    fit_method = 'dynesty' #'mcmc'
    n_steps = 40000 
    maxiter = 200000
    # n_live = 500
    # fit_mode:
    #  'full': all parameters are fitted with Dynesty, 
    #  'with_nnls': the abundance coefficients are derived with a least square algorithm, the rest are fitted with dynesty
    fit_mode = 'with_nnls' #'full'
    do_fit = True
    fit_two_zones = False
    do_plot_init = not(do_fit)

    spec_path_lst = []
    distance_lst = []
    T_star_lst = [] #K
    r_star_lst = [] #R_Sun
    lum_star_lst = [] #L_Sun
    output_path_lst = []
    wl_filter_lst = []

    # Select data to fit
    ######################

    #datadir = '/Users/jvarga/Data/YSO_spectra/JWST_MIRI/plots/'
    datadir = '/allegro6/matisse/varga/YSO_spectra/JWST_MIRI/plots/'
    #datadir = '/allegro6/matisse/varga/YSO_spectra/JWST_MIRI/DQ_Tau/'
    
    #outputdir1 = '/Users/jvarga/Data/YSO_spectra/JWST_MIRI/Uniform_Red_MINDS_TTauri_1224/'
    outputdir1 = '/allegro6/matisse/varga/YSO_spectra/JWST_MIRI/Uniform_Red_MINDS_TTauri_1224/GRF_new/'
    outputdir2 = '/allegro6/matisse/varga/YSO_spectra/JWST_MIRI/DQ_Tau/'
    outputdir3 = '/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/'
    outputdir4 = '/Users/jvarga/Data/YSO_spectra/modeling/'
    # filepath, distance(pc), T_eff(K), L_star(L_Sun), r_star(R_Sun),outputdir,list of wavelength ranges to be excluded
    spec_data =[
    [datadir+'/DGTau_MINDS_MIRI.dat',125.00,4000.0,0.9,np.nan,outputdir1,[] ],# [[13.3,14.2]] ],
    [datadir+'/XXCha_MINDS_MIRI.dat', 194.64,3340.0,0.29,np.nan,outputdir1,[[13.3,14.2],[14.8,15.1]] ],
    [datadir+'/AATau_MINDS_MIRI.dat',137.72,3762.0,0.72,np.nan,outputdir1,[[13.5,14.2]] ],
    [datadir+'/BPTau_MINDS_MIRI.dat',128.28,3777.0,0.83,np.nan,outputdir1,[[24.5,35.0]] ],
    [datadir+'/CXTau_MINDS_MIRI.dat',126.74,3487.0,0.34,np.nan,outputdir1,[] ],
    [datadir+'/CYTau_MINDS_MIRI.dat',124.35,3516.0,0.37,np.nan,outputdir1,[[13.5,14.2]] ],
    [datadir+'/DFTau_MINDS_MIRI.dat',176.45,3900.0,3.89,np.nan,outputdir1,[[6.2,7.2],[13.3,14.2]]],
    [datadir+'/DLTau_MINDS_MIRI.dat',159.53,4276.0,1.47,np.nan,outputdir1,[[13.3,14.2]] ],
    [datadir+'/DMTau_MINDS_MIRI.dat',144.8, 3715.0,0.16,np.nan,outputdir1,[] ], #Lopez-Martinez+2015
    [datadir+'/DNTau_MINDS_MIRI.dat',127.29,3806.0,0.69,np.nan,outputdir1,[] ],
    [datadir+'/DRTau_MINDS_MIRI.dat',186.98,4202.0,3.71,np.nan,outputdir1,[[13.5,14.2]] ],
    [datadir+'/FTTau_MINDS_MIRI.dat',129.96,3415.0,0.44,np.nan,outputdir1,[[13.3,14.2]] ],
    [datadir+'/GWLup_MINDS_MIRI.dat',155.20,3632.0,0.33,np.nan,outputdir1,[[13.5,14.2]] ],
    [datadir+'/IMLup_MINDS_MIRI.dat',153.81,4350.0,2.57,np.nan,outputdir1,[[13.3,14.2]] ],
    [datadir+'/LkCa15_MINDS_MIRI.dat',154.83,4276.0,1.12,np.nan,outputdir1,[] ],
    [datadir+'/PDS70_MINDS_MIRI.dat', 112.32,4138.0,0.38,np.nan,outputdir1,[] ],
    [datadir+'/RNO90_MINDS_MIRI.dat', 114.96,5662.0,5.7 ,np.nan,outputdir1,[[13.5,14.2]] ], #Salyk+2013
    [datadir+'/RWAur_MINDS_MIRI.dat', 150.0 ,4900.0,2.95,np.nan,outputdir1,[[4.8,6.6]] ], #Testi+2022
    [datadir+'/SYCha_MINDS_MIRI.dat', 180.78,4060.0,0.43,np.nan,outputdir1,[[13.5,14.2]] ],
    [datadir+'/Sz50_MINDS_MIRI.dat',  148.94,3400.0,0.41,np.nan,outputdir1,[[13.5,14.2]] ], #dist Bailer-Jones, others: Sartori 2003
    [datadir+'/Sz98_MINDS_MIRI.dat',  156.27,4060.0,1.51,np.nan,outputdir1,[[13.7,14.2]] ],
    [datadir+'/TWHya_MINDS_MIRI.dat',  59.96,4000.0,0.34,np.nan,outputdir1,[] ],
    [datadir+'/V1094Sco_MINDS_MIRI.dat',152.44,4205.0,1.15,np.nan,outputdir1,[[13.3,14.2],[24.2,35.0]] ],
    [datadir+'/VWCha_MINDS_MIRI.dat', 188.16,4060.0,1.62,np.nan,outputdir1,[[13.3,14.2]] ],
    [datadir+'/WAOph6_MINDS_MIRI.dat',122.53,4169.0,2.88,np.nan,outputdir1,[[6.4,6.7],[13.3,14.2]] ], #Andrews+2018 DSHARP
    [datadir+'/WXCha_MINDS_MIRI.dat', 189.3 ,3780.0,0.62,np.nan,outputdir1,[[6.4,6.7],[13.3,14.2]] ], #Testi+2022
    
    ]
    [
    [datadir+'/postprocess_spectrum_rb_epoch1.txt',196.0,3500.0,1.72,np.nan,outputdir2,[]], #dist: GAIA EDR3, Lum: Akesson+2019: 1.55, Teff: Kraus+2009: 3850.0
    [datadir+'/postprocess_spectrum_rb_epoch2.txt',196.0,3500.0,1.42,np.nan,outputdir2,[]],
    [datadir+'/postprocess_spectrum_rb_epoch3.txt',196.0,3500.0,1.47,np.nan,outputdir2,[]],
    [datadir+'/postprocess_spectrum_rb_epoch4.txt',196.0,3500.0,1.67,np.nan,outputdir2,[]],
    ]
    [
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J181030_smoothed_rebinned.dat',101.0,5900.0,3.8,np.nan,outputdir3], #HD 166191
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J010942_smoothed_rebinned.dat',184.1,5500.0,0.8,1.0,outputdir3], #RZ Psc, Su+2023
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J060513_smoothed_rebinned.dat',262.1,5660.0,0.94 ,np.nan,outputdir3], #TYC 5940-1510-1, Moor+2021
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J061103_smoothed_rebinned.dat',184.1,5350.0,0.65,np.nan,outputdir3], #TYC 8105-370-1, Moor+2021
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J060917_smoothed_rebinned.dat',348.3,5409.0,0.493,0.8,outputdir3],  #Moor+2024
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J071206_smoothed_rebinned.dat',174.3,4682.0,0.284,0.84,outputdir3],  #Moor+2024
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J092521_smoothed_rebinned.dat',98.9, 4513.0,0.224,0.81,outputdir3],  #Moor+2024
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J104416_smoothed_rebinned.dat',199.1,5241.0,0.406,0.77,outputdir3],  #Moor+2024
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J204315_smoothed_rebinned.dat',117.5,4642.0,0.207,0.7,outputdir3],  #Moor+2024
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J230112_smoothed_rebinned.dat',158.7,5300.0,0.65,0.96,outputdir3], #TYC 8830-410-1, Melis+2021
    ['/Users/jvarga/Data/YSO_spectra/JWST_MIRI/edd_miri_spectra/specfit/miri_J121334_smoothed_rebinned.dat',242.1,6350.0,2.55,np.nan,outputdir3], #TYC 4946-1106-1, Moor+2021
    ]

    [
    ['/Users/jvarga/Data/MATISSE/modeling/T_Cha/T_Cha_MIRI_2025_March.dat',102.7,5600.0,1.6,1.3,outputdir4],
    ]
    # do not fit:
    #[datadir+'/IRS46_MINDS_MIRI.dat',], #sil abs.
    # [datadir+'/SR21_MINDS_MIRI.dat',  136.43,4365.0,3.8 ,np.nan], #Natta+2006, PAH

    for item in spec_data:
        spec_path_lst.append(item[0])
        distance_lst.append(item[1])
        T_eff = item[2]
        Lum = item[3]
        T_star_lst.append(T_eff)
        lum_star_lst.append(Lum)
        try:
            r_star = np.sqrt(Lum * 3.828e26 / (4.0*np.pi*sigma_SB*T_eff**4))/7.1492e7/9.96
        except ZeroDivisionError as e:
            r_star = 0.001
        r_star_lst.append(r_star)
        output_path_lst.append(item[5])
        wl_filter_lst.append(item[6])
        # print(T_eff,r_star,Lum)
    #%%
    ################################################################
    # run modeling
    ################################################################

    for spec_path,outputdir,distance,T_star,r_star,lum_star,wl_filter in zip(spec_path_lst,output_path_lst,distance_lst,T_star_lst,r_star_lst,lum_star_lst,wl_filter_lst):
        target_data_label = os.path.basename(spec_path).split('.')[0]
        print(target_data_label)
        r_in = 0.069*np.sqrt(lum_star)
        if 'miri_J' in spec_path: #in these spectra the stellar spectrum is subtracted
            T_star = 0.0001
            r_star = 0.0001

        ################################################################
        # read input spectrum and opacity files
        ################################################################
        #read input spectrum
        if 'DQ_Tau' not in spec_path:
            wl,fluxdata,fluxerr_tmp = np.loadtxt(spec_path, comments="#", skiprows=1, usecols=(0,1,2), unpack=True)
        else:
            wl_full,tmp1,fluxerr_full,fluxdata_full,tmp2 = np.loadtxt(spec_path, comments="#", skiprows=1, usecols=(0,1,2,3,4), unpack=True)
            fluxdata, bin_edges, binnumber = stats.binned_statistic(wl_full, fluxdata_full,
                statistic=np.nanmedian, bins=500)
            fluxerr_tmp, bin_edges, binnumber = stats.binned_statistic(wl_full, fluxerr_full,
                statistic=np.nanmedian, bins=500)
            #flux_std, bin_edges, binnumber = stats.binned_statistic(wl_full, fluxdata_full,
            #    statistic=np.nanstd, bins=500)
            bin_width = (bin_edges[1] - bin_edges[0])
            wl = bin_edges[1:] - bin_width/2
            # print(fluxdata.shape)

        # if 'miri_J' in spec_path:
        #     outputdir = outputdir3
        # if 'T_Cha' in spec_path:
        #     outputdir = outputdir4

        if 'miri_J060917' in spec_path:
            wl_limits_up = 23.0
        else:
            wl_limits_up = wl_limits[1]
        wl_idx = np.logical_and(wl>=wl_limits[0],wl<=wl_limits_up)
        wl = wl[wl_idx]
        fluxdata = fluxdata[wl_idx]
        fluxerr_tmp = fluxerr_tmp[wl_idx]
        fluxerr = 0.0*fluxdata+np.nanmedian(fluxerr_tmp)

        # apply wl filter
        wl_idx = np.logical_and(wl>20.0,wl<=0.0) #all False
        #print('initial',wl)
        for wl_range in wl_filter:
            tmp_idx = np.logical_and(wl>wl_range[0],wl<=wl_range[1])
            wl_idx = np.logical_or(wl_idx,tmp_idx)
        fluxdata = fluxdata[~wl_idx]
        fluxerr = fluxerr[~wl_idx]
        wl=wl[~wl_idx]
        #print('final',wl)

        print('Read opacity files')
        if 'DHS' in run_name:
            opac_fname_list = opac_fname_list_DHS
            opac_dir = opac_dir_DHS
            opac_type =  'Q_abs' #'kappa_abs' #  
        if 'GRF' in run_name:
            opac_fname_list = opac_fname_list_GRF
            opac_dir = opac_dir_GRF
            opac_type =  'kappa_abs' # 'Q_abs' # 
        if 'JVmix' in run_name:
            opac_fname_list = opac_fname_list_JVmix
            opac_dir = opac_dir_JVmix
            opac_type =  'Q_abs' #'kappa_abs' # 
        kappa_arr = np.zeros((len(wl),len(opac_fname_list)))
        for i,fname in enumerate(opac_fname_list):
            print(fname,end=' ')
            if opac_type == 'Q_abs':
                N,gsize,gdens=np.loadtxt(opac_dir+'/'+fname,max_rows=1)
            n_comment_lines = 0
            with open(opac_dir+'/'+fname) as f:
                for line in f:
                    if line.startswith('#'):
                        n_comment_lines += 1
                    else:
                        break
            if opac_type == 'Q_abs':
                wl_Q,opac_data = np.loadtxt(opac_dir+'/'+fname, comments="#", skiprows=n_comment_lines+2, usecols=(0,1), unpack=True)
                kappa_arr[:,i] = np.interp(wl, wl_Q, 0.1* opac_data* 3.0 / (4.0 * gsize * 1e-4 * gdens)) #m^2/kg
            if opac_type == 'kappa_abs':
                # GRF opacity files: lambda[um], kappa_tot, kappa_abs, kappa_sca
                # optool opacity files: lambda[um]  kabs [cm^2/g]  ksca [cm^2/g]    g_asymmetry
                if n_comment_lines > 20:
                    usec = (0,1) # optool opacity file
                    print('optool opacity file')
                else: 
                    usec = (0,2) #GRF opacity files
                    print('GRF opacity file')
                wl_Q,opac_data = np.loadtxt(opac_dir+'/'+fname, comments="#", skiprows=n_comment_lines+2, usecols=usec, unpack=True)
                kappa_arr[:,i] = np.interp(wl, wl_Q, 0.1* opac_data) #convert to m^2/kg

        param_dic = {}
        kappa_label_list = []
        kappa_label_short_list = []
        grain_name_list = []
        grain_size_list = []
        
        if fit_mode == 'full':
            dust_coeff_free = True
        elif fit_mode == 'with_nnls':
            dust_coeff_free = False

        if fit_two_zones:
            param_dic['r_z'] = {'value':1.0,'limits':[r_in/5.0,15.0],'free':True}
            param_dic['n_z'] ={'value':2,'limits':[1,2],'free':False}
            zone_tags = ['_in','_out']
        else:
            param_dic['n_z'] ={'value':1,'limits':[1,2],'free':False}
            zone_tags = ['']

        for zone_tag in zone_tags:
            for Q_fname in opac_fname_list:
                label = os.path.splitext(Q_fname)[0]+zone_tag
                kappa_label_list.append(label)
                kappa_label_short_list.append((get_color_label(label))[1])
                # grain_name_list.append(re.sub(r'[0-9.]+', '', get_short_label(label)))
                # grain_size_list.append(re.sub("[^\d\.]", "", get_short_label(label)))
                grain_name_list.append((get_color_label(label))[1])
                grain_size_list.append((get_color_label(label))[2])
                param_dic[label]={'value':5e-4,'limits':[0.0,5e-2],'free':dust_coeff_free, #np.random.uniform(low=0.0, high=5e-2) # np.log10(2e-4) #'limits':[np.log10(1e-9),np.log10(5e-2)]
                                'grain_name':grain_name_list[-1] ,'grain_size':grain_size_list[-1],'mass_fraction':np.nan}
        param_dic['r_in'] =       {'value':r_in,'limits':[r_in/5.0,1.0],'free':True}
        param_dic['r_out'] =      {'value':20.0,'limits':[0.0,40.0],'free':False}
        param_dic['T_dust_in'] =  {'value':1200,'limits':[400.0,1700.0],'free':True}
        param_dic['q_dust'] =     {'value':-0.7,'limits':[-3.0,0.0],'free':True}
        param_dic['T_midplane_in'] = {'value':800,'limits':[100.0,1500.0],'free':True}
        param_dic['q_midplane'] = {'value':-1.4,'limits':[-3.0,0.0],'free':True}
        param_dic['rim_width'] =  {'value':0.01,'limits':[0.0,2.0],'free':True}
        param_dic['T_rim_in'] =   {'value':1400.0,'limits':[800.0,1800.0],'free':True}
        param_dic['q_rim'] =      {'value':0.0,'limits':[-3.0,0.0],'free':False}
        param_dic['T_star'] =     {'value':T_star,'limits':[2000.0,20000.0],'free':False}
        param_dic['r_star'] =     {'value':r_star,'limits':[0.0,100.0],'free':False} #in Solar radii
        param_dic['n_r'] =        {'value':100,'limits':[0.0,1.0],'free':False}
        param_dic['n_rim'] =      {'value':10,'limits':[0.0,1.0],'free':False}
        param_dic['gap_width'] =  {'value':0.0,'limits':[0.0,4.0],'free':False}
        param_dic['r_gap'] =      {'value':0.0,'limits':[0.0,15.0],'free':False}
        param_dic['dscale'] =     {'value':distance*parsec/au,'limits':[-1.0,1.0],'free':False}

        grain_name_list_unique,grain_size_counts = np.unique(grain_name_list,return_counts=True)

        # mass fractions
        n_free_params = 0
        free_param_labels = []
        free_param_values = []

        for label,param in param_dic.items():
            # print(label,param['free'])
            if param['free'] == True:
                n_free_params+=1
                free_param_labels.append(label)
                free_param_values.append(param['value'])

        ################################################################
        # plot initial model against data
        ################################################################

        output_path = outputdir+'/'+run_name+'_'+target_data_label +'_fit_init'+fit_method+'.png'
        model_flux = model_fn(wl,param_dic,[kappa_arr,kappa_label_list],fit_mode,fluxdata)
        flux_dust,flux_midplane,flux_star,flux_rim = get_component_fluxes(wl,param_dic,[kappa_arr,kappa_label_list],fit_mode,fluxdata)
        model_fluxes = [model_flux,np.nansum(flux_dust,axis=1),flux_midplane,flux_star,flux_rim,flux_dust]
        chi2_red = -2.0*lnlike(free_param_values, wl, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list],fit_mode])
        if do_plot_init:
            plot_title = target_data_label.replace('_MINDS_MIRI','')+r' init, $\chi^2_\mathrm{red}$ = %.2f'%(chi2_red)
            plot_fit(output_path,wl,fluxdata,model_fluxes,kappa_label_list,plot_title,param_dic,fit_mode)

        print('Mass fractions (%)')
        surfdens = 0.0
        if fit_mode == 'with_nnls':
            for kappa_label in kappa_label_list:
                #surfdens += 10.0**param_dic[kappa_label]['value']
                surfdens += param_dic[kappa_label]['value']
            for kappa_label in kappa_label_list:
                # mf = 10.0**param_dic[kappa_label]['value']/surfdens*100.0
                mf = param_dic[kappa_label]['value']/surfdens*100.0
                print('%-38s %.3f'%(kappa_label,mf))


        ##%%
        ################################################################
        # setup fitter and run fit
        ################################################################
        if do_fit:
            if fit_method == 'mcmc':
                nwalkers = n_free_params * 2
                burnin = int(n_steps/2)

                # initial positions of the walkers
                pos = [] 
                for param in param_dic.values():
                    if param['free'] == True:
                        margin=0.001*(param['limits'][1]-param['limits'][0])
                        pos.append(np.random.uniform(low=param['limits'][0]+margin,
                            high=param['limits'][1]-margin,size=nwalkers))
                pos = np.transpose(np.array(pos))

                # run MCMC
                with mp.get_context('fork').Pool(int(mp.cpu_count()/2)) as pool:
                    sampler = emcee.EnsembleSampler(nwalkers, n_free_params, lnprob, 
                            args=[wl, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list]]],pool=pool)
                    sampler.run_mcmc(pos, n_steps, progress=True)
            elif fit_method == 'dynesty':
                # Nested sampling. #int(mp.cpu_count()/2)
                with dynesty.pool.Pool(2, lnlike, prior_transform, 
                        logl_args=[wl, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list],fit_mode]],
                        ptform_args=[param_dic,free_param_labels]) as pool:
                    sampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, 
                        n_free_params, pool = pool,bound='multi',
                        sample='rwalk') # bound='single')

                    # sampler =  dynesty.NestedSampler(pool.loglike, pool.prior_transform,
                                    # n_free_params, pool = pool,nlive=n_live)
                    #sampler = dynesty.NestedSampler(lnlike, prior_transform, n_free_params,
                    #    logl_args=[wl, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list],fit_mode]],
                    #    ptform_args=[param_dic,free_param_labels],nlive=n_live)
                    sampler.run_nested(dlogz_init=0.05, nlive_init=500, nlive_batch=100,maxiter=maxiter,
                                       checkpoint_file=outputdir+'/'+run_name+'_'+target_data_label +'_dynesty.save')
                                    # maxiter=2000,use_stop=False)

            ##%%
            ################################################################
            # save and plot fit results
            ################################################################

            # Plot posterior distributions (corner plots)
            if fit_method == 'mcmc':
                chi2_red = -2.0*sampler.get_log_prob(discard = burnin, flat=True)
                samples = sampler.get_chain(discard = burnin, flat = True)
                confidence_intervals = list(map(lambda v: (v[0], v[1]),
                    zip(*np.percentile(samples, [16, 84],axis=0))))
                idx = np.argmin(chi2_red)
                chi2r_fit = chi2_red[idx]
                print('Min chi2:',chi2r_fit)
                best_parameters = samples[idx]
            elif fit_method == 'dynesty':
                dy_result = sampler.results
                chi2_red = -2.0*dy_result.logl
                samples = dy_result.samples
                weights = dy_result.importance_weights()
                confidence_intervals = [dyfunc.quantile(samps, [0.16, 0.84], weights=weights)
                    for samps in samples.T]

                # Compute weighted mean and covariance.
                # mean, cov = dyfunc.mean_and_cov(samples, weights)
                # best_parameters = mean
                idx = np.argmin(chi2_red)
                chi2r_fit = chi2_red[idx]
                print('Min chi2:',chi2r_fit)
                best_parameters = samples[idx]
                
            for (pvalue,label,confidence_interval) in zip(best_parameters,free_param_labels,confidence_intervals):
                param_dic[label]['value'] = pvalue
                param_dic[label]['err_l'] = pvalue-confidence_interval[0]
                param_dic[label]['err_u'] = confidence_interval[1]-pvalue
                param_dic[label]['conf_l'] = confidence_interval[0]
                param_dic[label]['conf_u'] = confidence_interval[1]
         
            #     print(label,pvalue)
            # if fit_method == 'dynesty':
                # chi2r_fit = -2.0*lnlike(best_parameters, wl, fluxdata, fluxerr,[param_dic,free_param_labels,[kappa_arr,kappa_label_list],lnlike])

            model_flux = model_fn(wl,param_dic,[kappa_arr,kappa_label_list],fit_mode,fluxdata)
            flux_dust,flux_midplane,flux_star,flux_rim = get_component_fluxes(wl,param_dic,[kappa_arr,kappa_label_list],fit_mode,fluxdata)

            #save best-fit parameters 
            outfile_path = outputdir+'/'+run_name+'_'+target_data_label +'_fit_best_params.txt'
            with open(outfile_path, "w") as f:
                f.write("Chi2r : {}\n".format(chi2r_fit))
                f.write('Fit method : %s\n'%(fit_method))
                if fit_method == 'mcmc':
                    f.write('nsteps : %d\n'%(n_steps))
                    f.write('ndiscard : %d\n'%(burnin))
                    f.write('nwalkers : %d\n'%(nwalkers))
                elif fit_method == 'dynesty':
                #     f.write('nlive : %d\n'%(n_live))
                    f.write('niter : %d\n'%(dy_result['niter']))
                f.write("----------------\n")
                f.write("Dust mass fractions (%)\n")
                print("\nDust mass fractions (%)")
                surfdens = 0.0
                for kappa_label in kappa_label_list:
                    if fit_mode == 'full':
                        surfdens += 10.0**param_dic[kappa_label]['value']
                    elif fit_mode == 'with_nnls':
                        surfdens += param_dic[kappa_label]['value']
                for kappa_label in kappa_label_list:
                    if fit_mode == 'full':
                        val = 10.0**param_dic[kappa_label]['value']/surfdens*100.0
                        if param_dic[kappa_label]['free'] == True:
                            err_l = (10.0**param_dic[kappa_label]['value']-10.0**param_dic[kappa_label]['conf_l'])/surfdens*100.0
                            err_u = (10.0**param_dic[kappa_label]['conf_u']-10.0**param_dic[kappa_label]['value'])/surfdens*100.0
                            outstr = '%-40s %7.3f (-%7.3f/+%7.3f)'%(kappa_label,val,err_l,err_u)
                        else: 
                            outstr = '%-40s %7.3f'%(kappa_label,val)
                    elif fit_mode == 'with_nnls':
                        val = param_dic[kappa_label]['value']/surfdens*100.0
                        #err_l = (param_dic[kappa_label]['value']-param_dic[kappa_label]['conf_l'])/surfdens*100.0
                        #err_u = (param_dic[kappa_label]['conf_u']-param_dic[kappa_label]['value'])/surfdens*100.0
                        outstr = '%-40s %7.3f (-%7.3f/+%7.3f)'%(kappa_label,val,0,0) #err_l,err_u)
                    print(outstr)
                    f.write(outstr+'\n')
                f.write("----------------\n")
                f.write("Free parameters\n")
                print("\nFree parameters")
                i = 0
                for (pvalue,label,confidence_interval) in zip(best_parameters,free_param_labels,confidence_intervals):
                    outstr = '%s: %.10f (-%.10f/+%.10f)'%(label,pvalue,pvalue-confidence_interval[0],confidence_interval[1]-pvalue)
                    f.write(outstr+'\n')
                    print(outstr)
                    i+=1
                f.write("----------------\n")
                f.write("Fixed parameters\n")
                i = 0
                for key, param in param_dic.items():
                    if param['free'] == False:
                        f.write('%s: %.15f\n'%(key,param['value']))
                        i+=1
                f.write("----------------\n")
                f.write("Input spectrum\n")
                f.write('%s\n'%(spec_path))
                f.write("----------------\n")
                f.write("Input opacity files\n")
                for fname in opac_fname_list:
                    f.write('%s\n'%(fname))

            # plot best-fit model against data
            
            model_fluxes = [model_flux,np.nansum(flux_dust,axis=1),flux_midplane,flux_star,flux_rim,flux_dust]
            plot_title = target_data_label.replace('_MINDS_MIRI','')+r' best-fit, $\chi^2_\mathrm{red}$ = %.2f'%(chi2r_fit)
            output_path = outputdir+'/'+run_name+'_'+target_data_label +'_fit_'+fit_method+'.png'
            plot_fit(output_path,wl,fluxdata,model_fluxes,kappa_label_list,plot_title,param_dic,fit_mode)
            plot_title = target_data_label.replace('_MINDS_MIRI','')
            output_path = outputdir+'/'+run_name+'_'+target_data_label +'_fit_'+fit_method+'.pdf'
            plot_fit(output_path,wl,fluxdata,model_fluxes,kappa_label_list,plot_title,param_dic,fit_mode)

            # save data and model spectra
            outfile_path = outputdir+'/'+run_name+'_'+target_data_label +'_fit_model_spectra.txt'
            with open(outfile_path, "w") as f:
                f.write('# Wl(um) F_data(Jy) err_F_data(Jy)  F_tot(Jy) F_dust(Jy) F_midp(Jy) F_star(Jy)  F_rim(Jy) ')
                for kappa_label in kappa_label_list:
                    f.write('F_nu_'+(get_color_label(kappa_label))[1]+'(Jy) ')
                f.write('\n')
                for i in range(len(wl)):
                    f.write('%8.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f '%(wl[i], fluxdata[i], fluxerr[i],  model_flux[i],(np.nansum(flux_dust,axis=1))[i],flux_midplane[i],flux_star[i],flux_rim[i]))
                    for j in range(len(flux_dust[0,:])):
                        f.write('%10.5f '%flux_dust[i,j])
                    f.write('\n')

            ##%%
            ################################################################
            # corner plot
            ################################################################
            try:
                output_path = outputdir+'/'+run_name+'_'+target_data_label +'_'+fit_method+'_corner_plot.png'
                short_param_labels = []
                for label in free_param_labels:
                    short_param_labels.append((get_color_label(label))[2])
                if fit_method == 'mcmc':
                    fig = corner.corner(samples, labels=short_param_labels,quantiles=[0.16,0.5,0.84],
                                        show_titles=False,use_math_text=True,
                                        truths=best_parameters, bins=40,label_kwargs={'fontsize':10},
                                        range=[0.999]*n_free_params) #bins=50

                elif fit_method == 'dynesty':
                    fig, ax = dyplot.cornerplot(dy_result, color='blue', truths=best_parameters,
                                            truth_color='black', show_titles=True,labels=short_param_labels,
                                            max_n_ticks=3, quantiles=[0.16,0.50,0.84],
                                            use_math_text=True) #title_fmt='.2e'
                                            #fig=(fig, axes[:, :3]))
                fig.savefig(output_path,dpi=200)
                #plt.show()
                plt.close(fig)
            except ValueError as e:
                print('Cannot make corner plot.')

    print('Finished '+run_name)
print('EXTERMINATE')
# %%

################################################################
# read previous results and plot
################################################################
load_previous_results = False

if load_previous_results:
    # run_name = '18_DHS_flaterr_3gs_5-25um'
    run_name = '17_GRF_flaterr_6gs_5-25um'

    for spec_path,result_dir,distance,T_star,r_star,lum_star in zip(spec_path_lst,output_path_lst, distance_lst,T_star_lst,r_star_lst,lum_star_lst):
        target_data_label = os.path.basename(spec_path).split('.')[0]
        print(target_data_label)
        outputdir = result_dir

        model_path = result_dir+'/'+run_name+'_'+target_data_label +'_fit_model_spectra.txt'
        model_data = np.loadtxt(model_path) #,usecols=(0,1,2,3,4,5,6,7),unpack=True)
        
        f = open(model_path)
        header = f.readline()
        f.close()

        wl = model_data[:,0]
        fluxdata = model_data[:,1]
        model_flux = model_data[:,3]
        flux_dust_sum = model_data[:,4]
        flux_midplane = model_data[:,5]
        flux_star = model_data[:,6]
        flux_rim = model_data[:,7]
        flux_dust = model_data[:,8:]
       
        #read best-fit parameters 
        param_dic = {}
        kappa_label_list = []
        best_params_file_path = result_dir+'/'+run_name+'_'+target_data_label +'_fit_best_params.txt'
        free_param_start = False
        fix_param_start = False
        dust_param_start = False

        with open(best_params_file_path, "r") as f:
            for line in f:
                if line.startswith('--------'):
                    free_param_start = False
                    fix_param_start = False
                    dust_param_start = False
                if free_param_start:
                    # print((line.split())[0].replace(':',''))
                    # print(float((line.split())[1]))
                    # print(((line.split())[2]))
                    param_dic[(line.split())[0].replace(':','')] = {'value':float((line.split())[1]),'limits':[np.nan,np.nan],'free':True}
                if fix_param_start:
                    # print((line.split())[0].replace(':',''))
                    # print(float((line.split())[1]))
                    label = (line.split())[0].replace(':','')
                    # gs = re.sub("[^\d\.]", "",get_short_label(label))
                    # gn = re.sub(r'[0-9.]+', '', get_short_label(label))
                    gn = (get_color_label(label))[1]
                    gs = (get_color_label(label))[2]
                    param_dic[label] = {'value':float((line.split())[1]),'limits':[np.nan,np.nan],'free':False,
                                                    'grain_size':gs,'grain_name':gn}
                if dust_param_start:
                    kappa_label_list.append((line.split())[0].replace(':',''))
                if line.startswith('Free parameters'):
                    free_param_start = True
                if line.startswith('Fixed parameters'):
                    fix_param_start = True
                if line.startswith('Dust mass fractions'):
                    dust_param_start = True

        # plot best-fit model against data
        output_path = outputdir+'/'+run_name+'_'+target_data_label +'_fit_'+fit_method+'.pdf'
        print(output_path)
        model_fluxes = [model_flux,flux_dust_sum,flux_midplane,flux_star,flux_rim,flux_dust]
        plot_title = target_data_label.replace('_MINDS_MIRI','') #+' best-fit, $\chi^2_\mathrm{red}$ = %.2f'%(chi2r_fit)
        plot_fit(output_path,wl,fluxdata,model_fluxes,kappa_label_list,plot_title,param_dic,fit_mode)

# %%
    
opac_species_list_DHS_old = [\
'Q_Am_Mgolivine_Jae_DHS_f0.7_rv0.1.dat',
# 'Q_Am_MgFeolivine_Dor_DHS_f0.7_rv0.1.dat',
'Q_Am_Mgpyroxene_Dor_DHS_f0.7_rv0.1.dat',
# 'Q_Am_MgFepyroxene_Dor_DHS_f0.7_rv0.1.dat',
# 'Q_Am_Ca2Al2SiO7_Mu_DHS_f0.70_rv0.1.dat',
# 'Q_Am_CaMgAlsil_Mu_DHS_f0.70_rv0.1.dat',
  'Q_Silica_MH_DHS_f0.7_rv0.1.dat',
  'Q_Fo_Zeidler_DHS_f0.8_rv0.1.dat',
# 'Q_Fay_Fabian_DHS_f0.8_rv0.1.dat',
'Q_En_Zeidler_DHS_f0.8_rv0.1.dat',
# 'Q_cor_0.10um_dhs_0.70.dat',
# 'Q_Koike_Fo100_combined.dat',
# 'Q_Fors_aerosol.dat',
# 'Q_Fo_Sogawa_DHS_f1.0_rv0.1.dat',
#  'Q_En_Jaeger_DHS_f1.0_rv0.1.dat',
'Q_Ann_Silica_Fabian_DHS_f0.99_rv0.1.dat',
]
opac_fname_list_DHS_old=[]
