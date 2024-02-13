#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:00:14 2023

@author: sand-jrd


Test the three algorithms on real datas. 

"""

from GreeDS import GreeDS, GreeDSRDI, find_param
from mustRDI import mustardRDI, theoretical_lim, lcurve
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_crop_frames, frame_shift, cube_shift
from os import mkdir, chdir
from os.path import isdir, isfile
from mustard.utils import circle
from vip_hci.preproc import frame_pad, frame_shift, frame_crop

import glob
from os.path import isdir, isfile
from vip_hci.greedy import pca_it
from vip_hci.psfsub import pca
import numpy as np

from vip_hci.stats.distances import cube_distance

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

dic_star = {
'V_CI_Tau':{'name':'CI_Tau','ref':'SSVS_1321'},
'V_CQ_Tau':{'name':'CQ_Tau','ref':'TYC_1865-648-1'},
'V_CY_Tau':{'name':'CY_Tau','ref':'J04170622+2802326'},
'V_DL_Tau':{'name':'DL_Tau','ref':'GSC_01833-00780'},
'V_DM_Tau':{'name':'DM_Tau' ,'ref':'GSC_01270-01088'},
'V_DN_Tau':{'name':'DN_Tau' ,'ref':'UCAC4_570-011400'},
'V_DS_Tau':{'name':'DS_Tau' ,'ref':'UCAC4_600-015051'},
'V_GM_Aur':{'name':'GM_Aur' ,'ref':'J04551015+3021333'},
'HD_31648':{'name':'HD_31648','ref':'HD_282758'},
'V1366_Ori':{'name':'HD_34282','ref':'BD-10_1143'},
'HD_97048':{'name':'HD_97048' ,'ref':'CD-76_498'},
'HD_100453':{'name':'HD_100453' ,'ref':'HD_100541'},
'HD_100546':{'name':'HD_100546' ,'ref':'HD_101869'},
'HD_143006':{'name':'HD_143006' ,'ref':'BD-21_4234'},
'HD_163296':{'name':'HD_163296' ,'ref':'HD_313493'},
'HD_169142':{'name':'HD_169142' ,'ref':'HD_169141'},
'V_IP_Tau':{'name':'IP_Tau' ,'ref':'J04284090+2655414'},
'V_IQ_Tau':{'name':'IQ_Tau','ref':'J04284090+2655414'},
'LkCa_15':{'name':'LkCa_15' ,'ref':'TYC_1279-203-1'},
'LkHA_330':{'name':'LkHA_330' ,'ref':'J03471855+3152187'},
'HD_36112':{'name':'MWC_758' ,'ref':'HD_244395'},
'CPD-68_1894':{'name':'PDS_66' ,'ref':'TYC_9246-822-1'},
'V351_Ori':{'name':'PDS_201' ,'ref':'HD_290774'},
'CPD-36_6759':{'name':'SAO_206462' ,'ref':'HD_135985'},
'SR_20':{'name':'SR_20' ,'ref':'WMR2005_3-26'},
'V_SY_Cha':{'name':'SY_Cha' ,'ref':'J11044460-7706240'},
'V_SZ_Cha':{'name':'SZ_Cha' ,'ref':'UCAC2_589393'},
'V_V1094_Sco':{'name':'V1094_Sco' ,'ref':'TYC_7855-1179-1'},
'V_V1247_ori':{'name':'V1247_Ori' ,'ref':'HD_290737'}}

cubedir = "./20231030_Ks_starhopping/"


# %%

chdir("/Users/sand-jrd/Desktop/DISK")

cubedir = "./20231030_Ks_starhopping/"

for datadir in list((glob.glob(cubedir + "Target*LkCa_15*"))):
    
    print(datadir)
    datadir = datadir.replace("__", "_")
    star_name = datadir.split("_202")[0].replace("./20231030_Ks_starhopping/Target_", "")
    comment = (datadir.split(star_name)[1][1:-1]).replace("_", ", ")
    date = comment[0:10]
    
    savedir = "./"+star_name+" - "+comment+"/"
    if not isdir(savedir) : mkdir(savedir)
   
    # if isfile(savedir+"ipcaRDI.fits"):
    #     print("Pass " + star_name)
    #     continue
    
    cube = open_fits(datadir)
    angles = open_fits(datadir.replace("Target", "ParallacticAngle")) + 135.99 - 1.75
    frame = np.mean(cube, axis=0)
    if star_name in dic_star.keys() :
        ref_star=dic_star[star_name]['ref']
        real_name=dic_star[star_name]['name']
        file_name=star_name

    else :
        for simba in dic_star.keys(): 
            if star_name==dic_star[simba]['name']:
                ref_star=dic_star[simba]['ref']
                real_name=dic_star[simba]['name']
                file_name=star_name

    angles = open_fits(datadir.replace("Target", "ParallacticAngle")) + 135.99 - 1.75
    
    reffiles = glob.glob(cubedir + "*Reference_*"+ref_star+"*")
    reffile = reffiles[0]
    # if len(reffile)> 1: 
            # if date in rfile :
            #     reffile=rfile
                
    ref = open_fits(reffile)
    ref = open_fits(reffiles[0])
    for rfile in reffiles:           
        ref = np.concatenate((ref,open_fits(rfile)),axis=0) 
    
    # refdirs = list(reversed(glob.glob(cubedir + "Reference_*"+date+"*")))
    nb_ref = int(len(angles)*1)

    # refdirs = list(reversed(glob.glob(cubedir + "*/Reference_")))
    
    # if len(refdirs)==0:
    #     if star_name == "HD_36112" : refdirs = list([cubedir+"Reference_HD_244395_2020-12-26a_Ks"])
    #     if star_name =='CPD-68_1894': refdirs = list([cubedir+"Reference_HD_244395_2020-12-26a_Ks"])
    # ref = open_fits(refdirs[0])
                
    # ind = np.flip(np.argsort(pcc))
    # ref = (ref[ind])[:nb_ref]
    # pcc = (pcc[ind])[:nb_ref]

    # if len(refdirs)>1:
        
    #### Select ref frames

    # reffiles = glob.glob(cubedir + "*Reference_*")
    # ref = open_fits(reffiles[0])
    # shape = ref.shape[-1]
    # pup = circle((shape,shape),shape//2) -circle((shape,shape),10)

    # pcc =  np.array(np.abs(cube_distance(ref, frame,  dist="pearson", plot=False)))
    # for refdir in reffiles[1:]:
    #     ref_add = open_fits(refdir)
    #     pcc_add =  np.array(np.abs(cube_distance(ref_add, frame, mode='mask', mask=pup, dist="pearson", plot=False)))
    #     ref = np.concatenate((ref, ref_add),axis=0)
    #     pcc = np.concatenate((pcc, pcc_add),axis=0)
        
    #     ind = np.flip(np.argsort(pcc))
    #     ref = (ref[ind])[:nb_ref]
    #     pcc = (pcc[ind])[:nb_ref]

    # # #ref = cube_crop_frames(ref, cube.shape[-1], force=True)

    # write_fits(datadir.replace("Target", "Refcustom"), ref)
    # plt.plot(pcc, label=star_name)     
    
    ref_c = ref[:nb_ref]
    while len(ref_c)<nb_ref:
        ref = np.concatenate((ref,ref), axis=0)
        ref_c = ref[:nb_ref]

    # %% Param
    
    # For save
    
    # %%
    ## -- For algos
    pup_size=0
    
    l=5
    r=20
    r_start=1
    full_output=1
    
    # %% Processing
    
    
    if True:#isfile(savedir+"ipcaRDI") :
        res_GreeDS = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=0, full_output=full_output, RDI=ref_c, perfect=False)
        write_fits(savedir+"GreeDSRDI",res_GreeDS)
        with open(savedir+'param.txt', 'w') as f:
            f.write("l = "+str(l)+"\nr="+str(r)+"\nr_start="+str(r_start))
    else : res_GreeDS = open_fits(savedir+"GreeDSRDI")
    
    
    if False :#not isfile(savedir+"ipcaRDI") :
        res_must, values = mustardRDI(cube, angles, ref, pup_size, save="./", plot=False, percreg=0.5)#testdir+"/X_"+test_ID)
    # res_must, resall = lcurve(cube, angles, ref, pup_size, save="./")
        write_fits(savedir+"mustRDI",res_must)
    # write_fits(savedir+"mustRDI_all",np.array(resall))
    else : res_must = open_fits(savedir+"mustRDI")

    #res_must = open_fits(savedir+"mustRDI")
    

    if False :#not isfile(savedir+"ipcaRDI") :
        res_PCA = pca_it(cube, angles, thr=0, r_out=None)# mask_val=0, edge_blend='interp', interp_zeros=True, ker=1, regul=True)
        write_fits(savedir+"ipcaRDI",res_PCA)
    else: res_PCA = open_fits(savedir+"ipcaRDI")
    
    # %% Plots
    
    # shape=182
    # perX = 99.8
    # perL = 99
    # perD = 99
    
    # xticks_lab = np.array([-0.5, 0, 0.5])
    # yticks_lab = np.array([-0.5, 0, 0.5])
    # plsca = 0.01226
    # xticks = (xticks_lab/plsca) + shape/2
    # yticks = (yticks_lab/plsca) + shape/2
    
    
    # pup = circle((shape,shape),shape//2) -circle((shape,shape),10)
    
    # def genere_args(Tm, M, Dtype, must=False):
    #     if Dtype == "X": 
    #         arg = {"cmap":"magma"}
    #         per = perX
    #         vmin = np.percentile(M[np.where(M>0)], 10)
    #         Tm[np.where(Tm<=0)] = vmin
    #         M[np.where(M<=0)] = vmin
    #         Tm[np.where(Tm<=0)] = vmin
    #         arg["norm"]= LogNorm(vmin=vmin, vmax=np.percentile(M, per))
    #         print(vmin)
    #     elif Dtype == "L": 
    #         arg = {"cmap":"jet"}
    #         per = perL
    #         arg["vmax"]=np.percentile(Tm, per)
    #     else :
    #         arg = {"cmap":"seismic"}
    #         per = perD
    #         arg["vmax"]=np.percentile(M, per)
    #     arg["X"]=Tm
    #     return arg
    
    
    # fig = plt.figure("real-data " +star_name, figsize=(15,4))
    
    # X_must =  pup*np.sqrt(abs(frame_crop(res_must, shape)))
    # X_PCA =  pup*np.sqrt(frame_crop(res_PCA, shape).clip(min=0))
    
    # best_frame=-9
    # X_GreeDS =  pup*np.sqrt(abs(frame_crop(res_GreeDS[best_frame], shape)))
    # im_ratio = 1
    
    # plt.subplot(1,3,1)
    # heatmap = plt.imshow(**genere_args(X_must, X_must, "X", True))
    # plt.title("MUSTAR ARDI")
    # plt.gca().invert_yaxis()
    # plt.xticks(xticks, labels = xticks_lab)
    # plt.yticks(yticks, labels = yticks_lab)
    # plt.colorbar(fraction=0.046*im_ratio, pad=0.04, shrink=0.9)
    
    # plt.subplot(1,3,3)
    # plt.imshow(**genere_args(X_PCA, X_PCA, "X", True))
    # plt.title("I-proj PCA with RDI")
    # plt.gca().invert_yaxis()
    # plt.xticks(xticks, labels = xticks_lab)
    # plt.yticks(yticks, labels = yticks_lab)
    # plt.colorbar(fraction=0.046*im_ratio, pad=0.04, shrink=0.9)
    
    # plt.subplot(1,3,2)
    # plt.imshow(**genere_args(X_GreeDS, X_GreeDS, "X", True))
    # plt.title("I-PCA with ARDI")
    # plt.gca().invert_yaxis()
    # plt.xticks(xticks, labels = xticks_lab)
    # plt.yticks(yticks, labels = yticks_lab)
    # plt.colorbar(fraction=0.046*im_ratio, pad=0.04, shrink=0.9)
    
    # plt.suptitle(real_name)
    # fig.subplots_adjust(right=0.95, left=0.03)
    # plt.savefig(savedir+"comp.png")
    # # cbar = fig.colorbar(heatmap, ax=[plt.subplot(1,3,1), plt.subplot(1,3,2), plt.subplot(1,3,3)], fraction=0.046*im_ratio, pad=0.04, shrink=0.9)
