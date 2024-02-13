"""
Created on Thu Nov 10 08:43:22 2022

   °______________________________°
   |                              |
   |         GreeDS - DEMO        |
   |______________________________|
   °                              °
  
GreeDS algorithm from Pairet etal 2020.
Basic implemented that works independently from MAYONNAISE.
* Kornia dependecy have been removed (depecated) 
* Added "r_start" option to improve results
* Added mode to use RDI as prior

@author: sand-jrd
"""

from GreeDS import GreeDS, find_optimal_iter, find_param, GreeDSRDI
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_crop_frames
import numpy as np
from os import  mkdir
from vip_hci.psfsub import pca, pca_annular
import matplotlib.pyplot as plt
from vip_hci.greedy import pca_it
from mustard.utils import circle
from vip_hci.stats.distances import cube_distance
from glob import glob
import vip_hci as vip
from vip_hci.fm import (confidence, cube_inject_companions, cube_planet_free, firstguess, mcmc_negfc_sampling, 
                            normalize_psf, show_corner_plot, show_walk_plot, speckle_noise_uncertainty)
from vip_hci.psfsub import median_sub, pca, pca_annular, pca_annulus, pca_grid
from vip_hci.metrics import contrast_curve, detection, significance, snr, snrmap, throughput
from vip_hci.var import fit_2dgaussian, frame_center


first=True
my_channel = 0
# %%  Load data

for bigdir in np.flip(glob("../../Archive PDS70/PDS70-neomayo/1100.C-0481D/")):
    for dir in glob(bigdir+"/*/"):

        plt.clf()
        if "res" in dir : pass
            
        cube = open_fits(dir+"cube.fits") # Must be one channel cube 
        angles = open_fits(dir+"angle.fits")
        psf = open_fits(dir+"psf.fits")

        # (optional) Reference frames. Add a data-driven prior using referene frames
        
        ref = None# np.concatenate([open_fits(dir+"ref_0.fits")[0][0:50], open_fits(dir+"ref_3.fits")[0][0:50], open_fits(dir+"ref_2.fits")[0][0:50]], axis=0) #or None
        # ref = open_fits(dir+"ref.fits")[my_channel]
        # %% Set parameters
        
        r = 10  # Iteration over PCA-rank
        l = 10  # Iteration per rank
        r_start  = 1 # PCA-rank to start iteration (good for faint signal)
        pup_size = 3 # Raduis of numerical mask to hide coro
        
        # Outputs (default 1) 
        full_output = 1 
        #  0/False -> only last estimation 
        #  1/True  -> every iter over r*l
        #  2       -> every iter over r
        #  3       -> every iter over l
        
        # (optional) Crop you cube 
        # crop_size = 256
        # cube = cube_crop_frames(cube, crop_size)
        # if ref is not None : ref  = cube_crop_frames(ref, crop_size)
        # save = name+"_chan_"+str(my_channel)

        # %% Plot ref Lib
        
        # ref_list = []
        # dists = ['pearson']
        # dist = dists[-1]
        
        # # plt.figure(dist)
        # # plt.title(dist)
        # nb_frame = len(cube)
        # masked_areas = circle(cube.shape[1:], 60)-circle(cube.shape[1:], 30) >0
            
        # # Ref for metric X
        # ref_select_man = []
        # for frame in cube:
        #     ref_select_man.append(np.abs(cube_distance(ref, frame, mode='mask', dist="pearson", mask=masked_areas, plot=False)))
        # ref_select_man = np.array(ref_select_man)
                
        # ref_select = np.mean(ref_select_man, axis=0)
        # ind = np.flip(np.argsort(ref_select))
        # ref_select = ref_select[ind]
        
        # plt.plot(ref_select, label=save)
        # if first :
        #     first=False
        #     plt.plot(nb_frame,ref_select[nb_frame], marker="o", markersize=6, color="black",  label="data cube size")
        # else : 
        #     plt.plot(nb_frame,ref_select[nb_frame], marker="o", markersize=6, color="black")

        
        # ref = ref[ind]
        # res = []
        # res1 = []
        # res2 = []
        # if first:
        #     first=False
        # else:
        #     for k in range (6,16,1):
        #         res.append(pca(cube, angles, ncomp=k, cube_ref=ref[ref_select>0.8]))
        #     #res2.append(pca_it(cube, angles, cube_ref=ref, strategy="RDI", ncomp=k))
        
        #     write_fits(dir+"PCA_RDI_"+save, np.array(res))
        #     #write_fits(dir+"PCA_RDI_Iproj_"+save, np.array(res2))
       
        
        #%%
        # param = "rank_"+str(5)+"_TO_"+str(15)+"_iter_"+str(2)+"_"

        # res = GreeDSRDI(cube, angles, ref=ref,r=15, l=2, r_start=6, pup=pup_size, full_output=full_output, returnL=False)
        # write_fits(dir+"RDI_iproj_"+param+save, res)

        # ref = ref[:len(angles)//2]

        # %% Greeds
        
        suuf = "new"
        param = "ratio_1.2_rank_"+str(r_start)+"_TO_"+str(r)+"_iter_"+str(l)+"_"

        # res = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output, RDI=ref, returnL=False, perfect=False)
        # write_fits(dir.replace("/","_")[34:]+"high_rank", res)
        # res = []
        # for k in range (1,15):
        #       res.append(pca(cube, angles, ncomp=k))
        # write_fits(dir.replace("/","_")[34:]+"PCA_rank1to15",np.array(res))

        # resX, indx = find_optimal_iter(res, plot=False, gtol=1e-2, noise_lim=(80), singal_lim="app", r=r, l=l, r_start=r_start, saveplot=False)
        # find_param(cube, angles, pup=pup_size, full_output=full_output, RDI=ref, returnL=False)
        
        # minortick = np.array(range(len(res)))
        # if l=="incr":
        #     majortick= []
        #     tmp = 0
        #     for k in range(0,r-r_start) : 
        #         majortick.append(tmp+k)
        #         tmp=tmp+k+1
        #     majortick = np.array(majortick)
        # else : 
        #     majortick = np.array(range(0,r-r_start))*l

        # majroticklab = ["rank "+str(k) for k in range(r_start, r)]    
        # fig = plt.figure('corr', (14,5))
        # ax = fig.add_subplot(111)
        # for ii in range(len(res)):
        #     plt.annotate(str(ii), xy=(ii,0), weight="bold", color="black", xytext=(-5, 0), textcoords='offset points')
        #     if ii==indx : plt.annotate("Recommanded\n"+str(ii), xy=(ii,0), weight="bold", color="red", xytext=(-5, 0), textcoords='offset points')
        # ax.set_xticks(minortick, minor=True) # labels=np.array(list(range(1,10,3))*10)
        # ax.tick_params(axis='x', which='minor', length=3, width=1, colors="gray", pad=1, labelsize=8)
        # ax.tick_params(axis='x', which='major', length=5, width=1, colors='r', labelrotation=30, labelsize=10)
        # ax.set_xticks(majortick,labels=majroticklab, minor=False)
        # plt.ylabel("frame index")
        # plt.ylim((0,0.5))
        # plt.tight_layout()
        # fig.savefig(dir+"ARDI_"+param+save+"_correspondences.png")
        # # Write results
        # write_fits(dir+"ARDI_"+param+save, res)
#%% SNR maps

        res = open_fits("../../IPCA-PDS70/IPCA reductions/"+dir.replace("/","_")[34:]+".fits")
        # DF_fit = fit_2dgaussian(res[5].clip(min=0), crop=True, cropsize=10, debug=True, full_output=True, cent=(135,135))
        shape = res.shape[1]
        # print(dir.replace("/","_")[34:])
        # print(np.float(DF_fit[1]["fwhm_y"]))
        # print(np.float(DF_fit[1]["fwhm_x"]))
        # print((np.float(DF_fit[1]["centroid_y"])-shape/2)*0.01226)
        # print((np.float(DF_fit[1]["centroid_x"])-shape/2)*0.01226)
        # print(np.float(DF_fit[1]["amplitude"]))
        # print(np.float(DF_fit[1]["theta"]))
        
        # plt.savefig("../../IPCA-PDS70/"+dir.replace("/","_")[34:]+"frame30.png")
        DF_fit = fit_2dgaussian(psf, crop=True, cropsize=9, debug=True, full_output=True)
        fwhm = np.mean([DF_fit[1]['fwhm_x'],DF_fit[1]['fwhm_y']])
        psfn = normalize_psf(psf, fwhm, size=19, imlib='ndimage-fourier')
        plsca = 0.01226
        
        # x = 0.0619043699294103
        # y = 0.0950709632546148
        # x=137 - shape/2
        # y=133 - shape/2
        x_pix = 136
        y_pix = 134
        amp = 1e-04

        # rad_fc =  np.sqrt(( x**2 + y**2 ))
        # theta_fc = -70
        # cubefc = cube_inject_companions(cube, psf_template=psfn, angle_list=angles, flevel=amp, plsc=plsca, 
        #                                 rad_dists=rad_fc, theta=theta_fc)
        
        # write_fits("cubetest", cubefc)
        
        # res = GreeDS(cubefc, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output, RDI=ref, returnL=False, perfect=False)
        # write_fits("../../IPCA-PDS70/"+dir.replace("/","_")[34:]+"IPCA_fakecomp1", res)
        # res = []
        # for k in range (1,3):
        #       res.append(pca(cubefc, angles, ncomp=k))
        # write_fits("../../IPCA-PDS70/"+dir.replace("/","_")[34:]+"PCA_fake_comp1",np.array(res))

        # resX, indx = find_optimal_iter(res, plot=False, gtol=1e-2, noise_lim=(80), singal_lim="app", r=r, l=l, r_start=r_start, saveplot=False)
        # find_param(cube, angles, pup=pup_size, full_output=full_output, RDI=ref, returnL=False)
        # print("Fake comp 2")
        # print(theta_fc)
        # print(rad_fc)
        # print(amp)
        
        r_0, theta_0, f_0 = firstguess(cube, angles, psfn, ncomp=1, planets_xy_coord=[[x_pix, y_pix]], 
                               fwhm=fwhm)
        
        plpar = [(r_0[0], theta_0[0], f_0[0])]
        print(dir.replace("/","_")[34:]+"\n First Guess\n "+str(r_0[0])+"\n"+str(theta_0[0])+"\n"+str(f_0[0])+"\n")
        cube_emp = cube_planet_free(plpar, cube, angles, psfn)
        
        conv_test, ac_c, ac_count_thr, check_maxgap = ('ac', 50, 1, 50)

        conv_params = {'conv_test': conv_test,
                       'ac_c': ac_c,
                       'ac_count_thr': ac_count_thr,
                       'check_maxgap': check_maxgap}
        from multiprocessing import cpu_count

        nwalkers, itermin, itermax = (100, 200, 500)
        obs_params = {'psfn': psfn,
                      'fwhm': fwhm}

        algo_params = {'algo': pca,
                       'ncomp': 1}

        mcmc_params = {'nwalkers': nwalkers,
                   'niteration_min': itermin,
                   'niteration_limit': itermax,
                   'bounds': None,
                   'nproc': cpu_count()//2}
        initial_state = np.array([r_0[0], theta_0[0], f_0[0]])
        mu_sigma=True
        aperture_radius=2
        
        negfc_params = {'mu_sigma': mu_sigma,
                        'aperture_radius': aperture_radius}

#%%
        chain = mcmc_negfc_sampling(cube, angles, **obs_params, **algo_params, **negfc_params, 
                            initial_state=initial_state, **mcmc_params, **conv_params,
                            display=True, verbosity=0, save=False, output_dir='./')
#%%
        import pickle
        output = {'chain':chain}
        with open("../../IPCA-PDS70/"+dir.replace("/","_")[34:]+"negfc", 'wb') as fileSave:
            pickle.dump(output, fileSave)
        burnin = 0.3
        isamples_flat = chain[:, int(chain.shape[1]//(1/burnin)):, :].reshape((-1,3))
        val_max, conf = confidence(isamples_flat, cfd=68, gaussian_fit=False, verbose=True, save=False, ndig=1, title=True)
        pl_par = (val_max['r'],val_max['theta'],val_max['f'])
        print("\n\n\n\n\n  \t\tPLANETE PARAMETER\n\n\n"+pl_par+"\n\n\n\n")
        
        algo_options={'ncomp':1, 'annulus_width':4*fwhm}
        speckle_res = speckle_noise_uncertainty(cube, pl_par, np.linspace(0,359,360), angles, pca_annulus, 
                                                psfn, fwhm, aperture_radius=2, fmerit='sum', 
                                                algo_options=algo_options, transmission=None, mu_sigma=None, 
                                                wedge=None, weights=None, force_rPA=False, nproc=None, 
                                                simplex_options=None, bins=None, save=False, output=None, 
                                                verbose=True, full_output=True, plot=True)
        
        output = {'speckle_res':speckle_res}
        with open("../../IPCA-PDS70/"+dir.replace("/","_")[34:]+"speckle_noise_uncertainty", 'wb') as fileSave:
                pickle.dump(output, fileSave)

#%%

        # suuf = "new"
        # param = "rank_"+str(r_start)+"_TO_"+str(r)+"_iter_"+str(l)+"_"

        # res = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output, returnL=False)
        # resX, indx = find_optimal_iter(res, plot=False, gtol=1e-2, noise_lim=(80), singal_lim="app", r=r, l=l, r_start=r_start, saveplot=False)
        # # find_param(cube, angles, pup=pup_size, full_output=full_output, RDI=ref, returnL=False)
        
        # minortick = np.array(range(len(res)))
        # if l=="incr":
        #     majortick= []
        #     tmp = 0
        #     for k in range(0,r-r_start) : 
        #         majortick.append(tmp+k)
        #         tmp=tmp+k+1
        #     majortick = np.array(majortick)
        # else : 
        #     majortick = np.array(range(0,r-r_start))*l

        # majroticklab = ["rank "+str(k) for k in range(r_start, r)]    
        # fig = plt.figure('corr', (14,5))
        # ax = fig.add_subplot(111)
        # for ii in range(len(res)):
        #     plt.annotate(str(ii), xy=(ii,0), weight="bold", color="black", xytext=(-5, 0), textcoords='offset points')
        #     if ii==indx : plt.annotate("Recommanded\n"+str(ii), xy=(ii,0), weight="bold", color="red", xytext=(-5, 0), textcoords='offset points')
        # ax.set_xticks(minortick, minor=True) # labels=np.array(list(range(1,10,3))*10)
        # ax.tick_params(axis='x', which='minor', length=3, width=1, colors="gray", pad=1, labelsize=8)
        # ax.tick_params(axis='x', which='major', length=5, width=1, colors='r', labelrotation=30, labelsize=10)
        # ax.set_xticks(majortick,labels=majroticklab, minor=False)
        # plt.ylabel("frame index")
        # plt.ylim((0,0.5))
        # plt.tight_layout()
        # fig.savefig(dir+"ADI_"+param+save+"_correspondences.png")
        # # Write results
        # write_fits(dir+"ADI_"+param+save, res)
        

