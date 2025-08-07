#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import emcee
import corner
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import multiprocessing as mp
from scipy.optimize import nnls
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.ticker as plticker

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# plot style configuration
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
matplotlib.rcParams.update({'font.size': 13})

################################################################
# function definitions
################################################################

h = 6.62606957e-34; #(J s)
kboltz = 1.3806488e-23; #(J K-1)
c0 = 299792458.0; #(m s-1)
au = 1.495978707e11 # (m)
rsun = 6.9634e8 # (m)
parsec = 3.0857e16 # (m)
sigma_SB = 5.67037e-8 #(W m−2 K−4)

#Planck's law 
def B_nu(T,nu):
    #return 2.0*h*nu**3.0/(c0**2.0)/(np.exp(h*nu/(kboltz*T))-1)  #[W sr-1 m-2 Hz-1]
    return 2.0*h*nu[:,None]**3.0/(c0**2.0)/(np.exp(h/kboltz*np.outer(nu,1.0/T))-1)
      
def get_component_fluxes(wl,param_dic,kappa,fit_mode,fluxdata):

    kappa_arr,kappa_label_list = kappa
    # construct a radial grid, radius in au
    radius_rim = np.logspace(np.log10(param_dic['r_in']['value']),
        np.log10(param_dic['r_in']['value']+param_dic['rim_width']['value']),param_dic['n_rim']['value'], endpoint='true')
    radius = np.logspace(np.log10(param_dic['r_in']['value']), #+param_dic['rim_width']['value']),
        np.log10(param_dic['r_out']['value']),param_dic['n_r']['value'], endpoint='true')
    # r_gap_in = param_dic['r_gap']['value']-param_dic['gap_width']['value']/2.0
    # r_gap_out = param_dic['r_gap']['value']+param_dic['gap_width']['value']/2.0
    # idx =  np.logical_or(radius_full<r_gap_in,radius_full>r_gap_out)
    # radius = radius_full[idx]

    # temperature power laws
    tprofile_dust = (param_dic['T_dust_in']['value']*(radius/radius[0])**param_dic['q_dust']['value'])    
    tprofile_midplane = (param_dic['T_midplane_in']['value']*(radius/radius[0])**param_dic['q_midplane']['value'])
    tprofile_rim = (param_dic['T_rim_in']['value']*(radius_rim/radius_rim[0])**param_dic['q_rim']['value'])
    
    freq = (c0*1e6)/wl

    # dust optical depth
    if fit_mode == 'full':
        tau_sum = wl*0.0
        for i,kappa_label in enumerate(kappa_label_list):
            tau_sum += kappa_arr[:,i] * 10.0**param_dic[kappa_label]['value'] #param_dic['surfdens_dust']['value']
        tau_arr = np.transpose(np.broadcast_to(tau_sum,(len(radius),len(wl))))
            
    I_nu_map_rim = B_nu(tprofile_rim,freq)
    I_nu_map_midplane = B_nu(tprofile_midplane,freq) 
    # print(I_nu_map_midplane.shape) #(441, 100) 

    # now integrate over the disk surface to get the flux density
    flux_midplane = 1e26/param_dic['dscale']['value']**2*np.trapz(2*np.pi*radius[None,:]*I_nu_map_midplane, radius[None,:],axis=-1)
    flux_star = np.ravel(1e26/param_dic['dscale']['value']**2*(param_dic['r_star']['value']*rsun/au)**2*np.pi*B_nu(param_dic['T_star']['value'],freq))
    flux_rim = 1e26/param_dic['dscale']['value']**2*np.trapz(2*np.pi*radius_rim[None,:]*I_nu_map_rim, radius_rim[None,:],axis=-1)

    if param_dic['n_z']['value'] == 2:
        radius1 = radius[radius < param_dic['r_z']['value']]
        radius2 = radius[radius >= param_dic['r_z']['value']]
        tprofile_dust1 = tprofile_dust[radius < param_dic['r_z']['value']]
        tprofile_dust2 = tprofile_dust[radius >= param_dic['r_z']['value']]
    else:
        radius1 = radius 
        tprofile_dust1 = tprofile_dust
    if fit_mode == 'full':
        # 2 zones not implemented in 'full' mode
        I_nu_map_dust = tau_arr*(B_nu(tprofile_dust,freq)) #(1.0-np.exp(-tau_arr))
        flux_dust = 1e26/param_dic['dscale']['value']**2*np.trapz(2*np.pi*radius[None,:]*I_nu_map_dust, radius[None,:],axis=-1)
    else: #fit_mode: 'with_nnls'
        source_fn1 = (1e26/param_dic['dscale']['value']**2*np.trapz(2*np.pi*radius1[None,:]*B_nu(tprofile_dust1,freq), radius1[None,:],axis=-1))
        if param_dic['n_z']['value'] == 2:
            source_fn2 = (1e26/param_dic['dscale']['value']**2*np.trapz(2*np.pi*radius2[None,:]*B_nu(tprofile_dust2,freq), radius2[None,:],axis=-1))
            dust_arr = np.concatenate(((kappa_arr*source_fn1[:,None]),(kappa_arr*source_fn2[:,None])),axis=1)
        else: 
            dust_arr = kappa_arr*source_fn1[:,None]
        try:
            dust_coeffs = nnls(dust_arr ,fluxdata-flux_star-flux_rim-flux_midplane)[0]
        except RuntimeError:
            dust_coeffs = np.ones((len(kappa_label_list)))   
        except ValueError:
            dust_coeffs = np.ones((len(kappa_label_list))) 
        for i,kappa_label in enumerate(kappa_label_list):
            param_dic[kappa_label]['value'] = dust_coeffs[i]
        flux_dust = dust_arr*dust_coeffs[None,:] #np.nansum( ,axis=1)

    return flux_dust,flux_midplane,flux_star,flux_rim

def model_fn(wl,param_dic,kappa,fit_mode,fluxdata):
    flux_dust,flux_midplane,flux_star,flux_rim = get_component_fluxes(wl,param_dic,kappa,fit_mode,fluxdata)
    # print(flux_dust+flux_midplane+flux_star+flux_rim)
    return np.nansum(flux_dust,axis=1)+flux_midplane+flux_star+flux_rim

# parametrisation (for dynesty)
def prior_transform(uniform_samples,param_dic,free_param_labels):
    """The transformation for uniform sampled values to the
    uniform parameter space."""
    priors = np.array([(param_dic[key]['limits'][0], param_dic[key]['limits'][1]) for key in free_param_labels])
    return priors[:, 0] + (priors[:, 1] - priors[:, 0])*uniform_samples

def lnlike(free_param_values, wl, fluxdata, fluxerr,other_args):
    param_dic,free_param_labels,kappa,fit_mode = other_args
    for (pvalue,label) in zip(free_param_values,free_param_labels):
        param_dic[label]['value'] = pvalue

    model = model_fn(wl,param_dic,kappa,fit_mode,fluxdata)
    #idx = (model-fluxdata) > 0.0
    #if np.any(idx):
    #    chi2_1 = np.nansum((fluxdata[idx]-model[idx])**2/((fluxerr[idx]/10.0)**2))
    #else:
    #    chi2_1 = 0.0
    #if np.any(~idx):
    #    chi2_2 = np.nansum((fluxdata[~idx]-model[~idx])**2/(fluxerr[~idx]**2) )
    #else:
    #    chi2_2 = 0.0
    #return -0.5*((chi2_1+chi2_2)/len(fluxdata))
    return -0.5*(np.nansum((fluxdata-model)**2/(fluxerr**2) )/len(fluxdata))

def lnprob(free_param_values, wl, fluxdata, fluxerr,other_args):
    param_dic,free_param_labels,kappa = other_args
    for (pvalue,label) in zip(free_param_values,free_param_labels):
        low, up =  param_dic[label]['limits']
        #print(label,low ,pvalue , up)
        if not low < pvalue < up:
            return -np.inf
        param_dic[label]['value'] = pvalue
        #print(label,pvalue ,param_dic[label]['value'])

    model = model_fn(wl,param_dic,kappa)
    # print(fluxdata-model)
    # print(fluxdata)
    # print(model)
    # print((np.sum((fluxdata-model)**2/(fluxerr**2) )/len(fluxdata)))
    return -0.5*(np.nansum((fluxdata-model)**2/(fluxerr**2) )/len(fluxdata))

def get_linestyle(kappa_label):
    if 'rv0.1' in kappa_label or '0.10um' in kappa_label or '0.1.' in kappa_label:
        linestyle = '--'
    elif 'rv1.0' in kappa_label or '1.00um' in kappa_label or '1.0.' in kappa_label:
        linestyle = (0, (5, 10)) #'loosely dashed'   
    elif 'rv2.0' in kappa_label or '2.00um' in kappa_label or '2.0.' in kappa_label:
        linestyle = (5, (10, 3)) #'long dash with offset'
    elif 'rv3.0' in kappa_label or '3.00um' in kappa_label or '3.0.' in kappa_label:
        linestyle = '-.'
    elif 'rv4.0' in kappa_label or '4.00um' in kappa_label or '4.0.' in kappa_label:
        linestyle = (0, (3, 5, 1, 5, 1, 5))  #'dashdotdotted'
    elif 'rv5.0' in kappa_label or '5.00um' in kappa_label or '5.0.' in kappa_label:
        linestyle = ':'
    else:
        linestyle = (0, (5, 10)) #'loosely dashed'       
    return linestyle

def get_color_label(kappa_label):
    color_table = [\
        ['Am_Mgolivine_Jae',r'Am. $\mathrm{Mg}_{2}\mathrm{SiO}_{4}$','blue'],
        ['Am_Mgol_Jae',r'Am. $\mathrm{Mg}_{2}\mathrm{SiO}_{4}$','blue'],
        ['MgOlivine',r'Am. $\mathrm{Mg}_{2}\mathrm{SiO}_{4}$','blue'],
        ['Am_MgFeolivine_Dor','Am. MgFe-olivine','royalblue'],
        ['Fay_Fabian','Fayalite','deepskyblue'],
        ['Am_Mgpyroxene_Dor',r'Am. MgSiO$_3$','darkorange'],
        ['Am_Mgpyr_Dor',r'Am. MgSiO$_3$','darkorange'],
        ['MgPyroxene',r'Am. MgSiO$_3$','darkorange'],
        ['MgFepyroxene_Dor','Am. MgFe-pyroxene','gold'],
        ['Fo_Zeidler', 'Forsterite','lime'],
        ['Fo_Suto', 'Forsterite','lime'],
        ['Fo_Sogawa', 'Forsterite', 'lime'],
        ['Koike_Fo100', 'Forsterite', 'lime'],
        ['Fors_aerosol', 'Forsterite', 'lime'],
        ['En_Zeidler','Enstatirte','orangered'],
        ['Ens_Zeidler','Enstatite','orangered'],
        ['En_Jaeger','Enstatite','orangered'],
        ['Ens_Jaeger','Enstatite','orangered'],
        ['_cor_','Corundum','fuchsia'],
        ['Am_Ca2Al2SiO7_Mu','Am. gehlenite','slategray'],
        ['Am_CaMgAlsil_Mu','Am. CaMgAl-silicate','darkseagreen'],
        ['_iron_','Iron','darkviolet'],
        ['_c-z_','Am. carbon','gray'],
        ['_sic_','SiC','rosybrown'], 
        ['Forsterite','Forsterite', 'lime'],
        ['Enstatite','Enstatite','orangered'],
        ['Olivine','Am. olivine','royalblue'],
        ['Pyroxene','Am. pyroxene','gold'],
        ['Am_Silica_Kit',r'Am. SiO$_2$','cyan'],
        ['Silica_MH',r'Am. SiO$_2$','cyan'],
        ['Ann_Silica',r'Ann. SiO$_2$','magenta'],
        ['Silica',r'Am. SiO$_2$','cyan'],
        ['qua',r'SiO$_2$','cyan']
        ]
    
    gs_table=[\
        ['rv0.1',0.1],
        ['rv1.0',1.0],
        ['rv1.5',1.5],
        ['rv2.0',2.0],
        ['rv3.0',3.0],
        ['rv4.0',4.0],
        ['rv5.0',5.0],
        ['rv6.0',6.0],
        ['0.10um',0.1],
        ['1.00um',1.0],
        ['1.50um',1.5],
        ['2.00um',2.0],
        ['3.00um',3.0],
        ['4.00um',4.0],
        ['5.00um',5.0],
        ['6.00um',6.0],
        ['0.1um',0.1],
        ['1.0um',1.0],
        ['1.5um',1.5],
        ['2.0um',2.0],
        ['3.0um',3.0],
        ['4.0um',4.0],
        ['5.0um',5.0],
        ['6.0um',6.0],    
        ['0.1.',0.1],
        ['1.0.',1.0],
        ['1.5.',1.5],
        ['2.0.',2.0],
        ['3.0.',3.0],
        ['4.0.',4.0],
        ['5.0.',5.0],
        ['6.0.',6.0],      
        ]
    color = 'black'
    short_label = kappa_label
    gs = 0.0
    for item in color_table:
        if item[0] in kappa_label:
            color = item[2]
            short_label = item[1]
            break
    for item in gs_table:
        if item[0] in kappa_label:
            gs = item[1]
            break
    return color,short_label,gs

#def get_short_label(kappa_label):
    #new_label = kappa_label.replace('Q_','').replace('DHS_','').replace('GRF_','').replace('.GRF','').replace('f0.7_','').replace('f0.8_','').replace('f0.99_','').replace('dhs_0.70','').replace('dhs_0.99','').replace('DHS_0.70','').replace('DHS_0.80','').replace('DHS_0.99','').replace('f1.0_','').replace('_rv','')
    #new_label = new_label.replace('Fo_Zeidler','Forsterite').replace('En_Zeidler','Enstatite').replace('En_Jaeger','Enstatite').replace('_Jae','').replace('_Dor','').replace('_MH','').replace('.Combined','')

def plot_fit(output_path,wl,fluxdata,model_fluxes,kappa_label_list,plot_title,param_dic,fit_mode):
    #fig, ((ax)) = plt.subplots(1, 1, sharey=False, sharex=False,figsize=(8,6))
    fig = plt.figure(figsize=(8,6.5)) #width, height
    gs1 = GridSpec(2, 1, height_ratios=[3.5,1],bottom=0.35,top=0.94,hspace=0.02)
    gs2 = GridSpec(1, 1,top=0.27)
    ax = fig.add_subplot(gs1[0])
    axr = fig.add_subplot(gs1[1])
    axb = fig.add_subplot(gs2[0])

    l0, = ax.plot(wl,fluxdata,'-',label='Data',lw=2,color='black') #'+'
    model_flux,flux_dust,flux_midplane,flux_star,flux_rim,dust_comp_flux = model_fluxes
    dust_comp_line_lst = []
    for i,kappa_label in enumerate(kappa_label_list):
        linestyle = get_linestyle(kappa_label)
        linecolor = (get_color_label(kappa_label))[0]
        line, = ax.plot(wl,dust_comp_flux[:,i],linestyle=linestyle,color=linecolor,alpha=0.8,label=(get_color_label(kappa_label))[1],lw=1)
        dust_comp_line_lst.append(line)
    l1, = ax.plot(wl,flux_dust,'-b',label='Dust',lw=1.5)
    l2, = ax.plot(wl,flux_midplane,'-g',label='Midplane',lw=1.5)
    l3, = ax.plot(wl,flux_rim,'-',label='Rim',lw=1.5,color='purple')
    l4, = ax.plot(wl,flux_star,'-',label='Star',lw=1.5,color='orange')
    l5, = ax.plot(wl,model_flux,'-r',label='Model',lw=2,alpha=0.7)
    #ax.set_ylim((ymin,ymax))
    #ax.set_xlabel('Wavelength ($\mu$m)')
    ax.set_ylabel('Flux density (Jy)')
    # ax.grid(which='minor',alpha=0.3)
    # ax.grid(which='major')
    ax.tick_params(top=True,bottom=True,left=True,right=True,direction="in",which="minor")
    ax.tick_params(top=True,bottom=True,left=True,right=True,direction="in",which="major")
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.minorticks_on( )
    ax.tick_params('x', labelbottom=False)
    # ax.tick_params(axis="x", which="minor", direction="in", 
    #                   top=True, bottom=True) #, labelbottom=False)
    plt.suptitle(plot_title)
    leg1 = ax.legend(handles=[l0,l5,l1,l2,l3,l4],framealpha=0.4,ncol=3,
        handletextpad=0.4,handlelength=2.0,prop={'size': 11.5},
        columnspacing=0.6) #loc='upper right'

    axr.plot(wl,0.0*wl,'--',color='gray',lw=1.5)
    axr.plot(wl,100.0*(fluxdata-model_flux)/fluxdata,'-k',label='Model',lw=2)
    axr.set_xlabel(r'Wavelength ($\mu$m)')
    axr.set_ylabel('Residual (%)')
    axr.xaxis.set_major_locator(loc)
    axr.tick_params(top=True,bottom=True,left=True,right=True,direction="in",which="minor")
    axr.tick_params(top=True,bottom=True,left=True,right=True,direction="in",which="major")
    axr.minorticks_on( )
    #leg2 = axr.legend(handles=dust_comp_line_lst,loc='lower center',bbox_to_anchor=(0.5, -0.07),
    #                bbox_transform=fig.transFigure, ncol=3,fontsize=12)
    
    #bar chart
    surfdens = 0.0
    for kappa_label in kappa_label_list:
        if fit_mode == 'full':
            surfdens += 10.0**param_dic[kappa_label]['value']
        elif fit_mode == 'with_nnls':
            surfdens += param_dic[kappa_label]['value']
    xlabels = []
    yvalues = []
    linestyle_lst = []
    barcolor_lst = []
    for kappa_label in kappa_label_list: 
        # tmp = param_dic[kappa_label]['grain_size']
        # if '.0' and tmp.endswith('.0'):
        #     tmp = tmp[:-len('.0')]
        if param_dic[kappa_label]['grain_size']<1.0:
            xlabels.append('%.1f'%(param_dic[kappa_label]['grain_size']))
        else:
            xlabels.append('%.0f'%(param_dic[kappa_label]['grain_size']))
        yvalues.append(param_dic[kappa_label]['value']/surfdens)
        linestyle_lst.append(get_linestyle(kappa_label))
        barcolor_lst.append((get_color_label(kappa_label))[0])
    bar_width=0.8
    axb.bar(range(len(xlabels)), yvalues,tick_label=xlabels,width=bar_width,
            log=True,color=barcolor_lst,alpha=1.0)#,edgecolor='black') # label=xl, color=bar_colors)
    ymin = 0.01
    for i in range(len(yvalues)):
        if yvalues[i] <= ymin:
            height = np.nan
        else:
            height = yvalues[i]-ymin

        axb.add_patch(Rectangle((i-bar_width/2.0, ymin), bar_width, height,
        fill=False,edgecolor='black', transform=axb.transData, clip_on=True,
        linestyle=linestyle_lst[i],linewidth=1.0))
    prev_name = ''
    first = True
    for i,(kappa_label) in enumerate(kappa_label_list):
        name = param_dic[kappa_label]['grain_name']
        name = (get_color_label(name))[1]

        if prev_name != name:
            axb.annotate(name,(i-bar_width/2.0,0.5),size=11)
            if first == True:
                first = False
            else:
               axb.plot([i-0.5,i-0.5],[ymin,1.0],'--',color='lightgray')
        prev_name = name

    axb.set_ylim(ymin,1.0)
    axb.set_ylabel('Mass fraction')
    axb.set_xlabel(r'Grain size ($\mu$m)')
    # print(xlabels, yvalues)

    plt.tight_layout() #pad=0.5)
    if 'pdf' in output_path:
        plt.savefig(output_path, dpi=200,bbox_inches="tight")
    else:
        plt.savefig(output_path, dpi=200,bbox_inches="tight")
    #plt.show()

# %%
