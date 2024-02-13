"""
Created on Thu Nov 10 08:43:22 2022

    ______________________________
                GreeDS - DEMO
    ______________________________

GreeDS algorithm from Pairet etal 2020.
Basic implemented that works independently from MAYONNAISE.
Nov 14 : Added r_start to improve results
Require the dependancy torch and kornia

@author: sand-jrd
"""

from GreeDS import GreeDS, GreeDSRDI, find_param
from mustRDI import mustardRDI, theoretical_lim, lcurve, circle
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_crop_frames, frame_shift, cube_shift, frame_pad, frame_crop
from os import mkdir, chdir
from os.path import isdir, isfile

import glob
from os.path import isdir, isfile
from vip_hci.greedy import pca_it
from vip_hci.psfsub import pca
import numpy as np

#%%

chdir("/Users/sand-jrd/Desktop/Data_challenge")
rand = True
## Load data

#dir = "your_directory"
prefd = "_sphere/"
Dir_output = "./test_cubes"+prefd
Dir_gtruth = "./res_cubes"+prefd
Dir_synth_disk = "./disk"+prefd
Dir_empty_cubes = "./empty_cubes"+prefd
Dir_aperture = "./aperture_disk"+prefd
tested_contrasts = open_fits("./test_cubes_sphere/tested_contrasts")

refdir = "./Ref_lib_sphere/"
cubedir = "./test_cubes_sphere/"
testdir = "./ARDI_perfect_ipca_estimate_all_sphere/"
testdir = "./mustard_personprofil_estimate_all_sphere/"
testdir = "./ARDI_ipca_mix_estimate_all_sphere/"

testdir_2 = "./ARDI_ipca2_estimate_all_sphere/"
testdir1 = "./ADI_ipca_shuffle_estimate_all_sphere/"
#testdir2 = "./RDI_pca_estimate_sphere/"
testdir2 = "./ARDI_ipca_shuffle_nooverlap_estimate_all_sphere/"
# testdir2 = "./ARDI_ipca_fake_estimate_all_sphere/"

testdir3 = "./RDI_ipca_shuffle_nooverlap_estimate_all_sphere/"
#testdir2 = "./mustard_lcurve2_estimate_all_sphere/"

#cubedir = "../dataset_rexpaco"
## Set parameters

r = 10   # Iteration over PCA-ranke
l = 10  # Iteration per rank
r_start = 1 # PCA-rank to start iteration (good for faint signal)
pup_size = 6 # Raduis of numerical mask to hide coro
ref_p = 0.5 # Percentage of reference frames

full_output = 1 # Return estimation at each iter (needed to search opti params) 
# If 0 -> only last estimation 
# if 1 -> every iter over r*l
# if 2 -> every iter over r
# if 3 -> every iter over d l

if not isdir(testdir) : mkdir(testdir)
txt_msg = "r = "+str(r)+", l = "+str(l)+", r_start = "+str(r_start)
with open(testdir + "/config.txt", 'w') as f:  f.write(txt_msg)

if not isdir(testdir1) : mkdir(testdir1)
txt_msg = "thr=0, r_out=None, mask_val=0, edge_blend='interp', interp_zeros=True, ker=1, regul=True"
with open(testdir1 + "/config.txt", 'w') as f:  f.write(txt_msg)

if not isdir(testdir2) : mkdir(testdir2)
txt_msg = "thr='auto', r_out=80, mask_val=0, edge_blend='interp', interp_zeros=True, ker=1, regul=True"
with open(testdir2 + "/config.txt", 'w') as f:  f.write(txt_msg)

if not isdir(testdir3) : mkdir(testdir3)
if not isdir(testdir_2) : mkdir(testdir_2)


datadir = cubedir+"test_1_1_0"
param_dict = [{'r':30, 'l':1, 'r_start':0}, {'r':20, 'l':20, 'r_start':0}, {'r':30, 'l':10, 'r_start':0},{'r':30, 'l':1, 'r_start':5}, {'r':20, 'l':20, 'r_start':5}, {'r':30, 'l':10, 'r_start':5}]
ref_p_l = [0.1, 0.5, 1, 2]

cube = open_fits(datadir+"/cube")
ref = open_fits(refdir+"/ref_1")
# ref_select = open_fits(refdir+"/ref_select_1").astype(bool)
# ref = ref[0,ref_select[0]]
angles = open_fits(datadir+"/angles")
# ref = cube_crop_frames(ref, cube.shape[-1], force=True)
# ii = 0
# for param in param_dict : 
#     for refp in ref_p_l : 
#         ref_c = ref[:int(cube.shape[0]*refp)]
#         res = GreeDS(cube, angles, **param, pup=pup_size, full_output=full_output, RDI=ref_c)
#         write_fits(testdir3+"X_110_"+str(ii)+"_"+str(refp), res)
#     ii+=1

for aa in range(0,3):
    listc = [0,1,2,3]
    listc.pop(aa)
    F=0
    print(listc)
    for ii in listc:
        
        ref = open_fits(refdir+"/ref_lib_"+str(int(ii)))[0]
        indx = np.random.rand(26)*len(ref)
        indx = indx.astype(int)
        if F == 0 : 
            print(F)
            F=1
            rand_ref=ref[indx]
            print(ii)
            print(F)

        else:
            rand_ref=np.concatenate((rand_ref,ref[indx]),axis=0)
            print(ii)
    
    np.random.shuffle(rand_ref)
    write_fits(refdir+"randref_nooverlap_"+str(aa), rand_ref)

#%%
rand = True

for datadir in list(reversed(glob.glob(cubedir + "*/"))):
    pref = "sphere_"

    test_ID = datadir.split("/")[-2][5:]
    size = 511
    disk_id = test_ID[2]
    cube_id = test_ID[0]
    cts_id = int(test_ID[-1])
    
    app        = open_fits(Dir_aperture+"app_disk_"+str(disk_id))
    disk = open_fits(Dir_synth_disk+ "res_disk_" + str(disk_id))
    if disk_id=='4' : 
        disk      = frame_pad(disk, size/disk.shape[0])
        #app       = frame_crop(app, size)
    else :
        disk      = frame_crop(disk, size)
        #app       = frame_crop(app, size)

    
    # if isfile(testdir2+"/X_"+test_ID+".fits"):
    #     print("Pass " + test_ID)
    #     continue

    flux = open_fits(Dir_empty_cubes+pref+"flux"+ "_" + str(cube_id))[0]

    X_res = disk*tested_contrasts[cts_id]

    edge = X_res.shape[-1]//2
    pup = circle(X_res.shape, edge)-circle(X_res.shape, 8)
        
    X_res = X_res*pup

    angles = open_fits(datadir+"/angles")
    cube = open_fits(datadir+"/cube")
    # cube = open_fits(refdir+"/ref_lib_"+str(int(test_ID[0])))[0][:len(angles)]
    # cube = open_fits(Dir_empty_cubes+pref+"cube"+ "_" + str(cube_id))[0]

    # cube = cube_crop_frames(cube, size, force=True)


    if rand : 
        ref = open_fits(refdir+"randref_nooverlap_"+str(cube_id))
    else : 
        ref = open_fits(refdir+"/ref_lib_"+str(int(test_ID[0])))[0]
        ref_select = open_fits(refdir+"/pcc_matrix_"+str(int(test_ID[0])))[0]
        
        ref_select = np.mean(ref_select, axis=0)
        ind = np.flip(np.argsort(ref_select))
        ref_select = ref_select[ind]
        ref = ref[ind]
    
    ref = cube_crop_frames(ref, cube.shape[-1], force=True)
    ref_c = ref[:int(cube.shape[0]*1)]
    
    # print("max coor = "+str(np.max(ref_select[0])))
    # print("min coor = "+str(np.max(ref_select[int(cube.shape[0]*1)])))

    # MUSTARD
    # res = mustardRDI(cube, angles, ref, pup_size, save="./")#testdir+"/X_"+test_ID)
    # res, rel = lcurve(cube, angles, ref, pup_size, save=testdir2+"/X_"+test_ID, res=X_res, factor=pup/(flux*np.sum(disk*app)))
    # write_fits(testdir2+"/X_"+test_ID, res)
    # write_fits(testdir2+"/X_"+test_ID+"_all", rel)

    # Find optimal param GreeDS
    # testdir2 = "./find_opti_GreeDS/"
    # find_param(cube, angles, noise_lim=150, singal_lim=(15,35), pup=pup_size, full_output=full_output, RDI=ref_c, returnL=False, savedir=testdir2+"/X_"+test_ID)

    # Limit Theorique
    # empty_cube = open_fits(dir_empty_cube+"/sphere_cube_"+test_ID[0])[0]
    # empty_cube = cube_crop_frames(empty_cube, cube.shape[-1], force=True)
    # res = theoretical_lim(cube, angles, empty_cube, ref_c)
    # write_fits(testdir2+"/X_"+test_ID, res)

    # # Greeds+ref
    res1 = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output, RDI=ref_c, perfect=False)
    write_fits(testdir2+"/X_"+test_ID, res1)
    
    # # Greeds+ref
    # res1 = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output, RDI=ref_c, perfect=True)
    # write_fits(testdir+"/X_"+test_ID, res1)
    

    # res1 = GreeDS(cube, angles, r=20, l=3, r_start=r_start, pup=pup_size, full_output=full_output, RDI=ref_c)
    # write_fits(testdir_2+"/X_"+test_ID, res1)
    
    # Greeds
    #X_est = open_fits("RDI_pca_estimate_511_sphere/X_"+test_ID)
    #res = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output)
   # write_fits(testdir1+"/X_"+test_ID, res)

    # Greeds RDI iteratif
    # res = GreeDSRDI(cube, angles, ref=ref, r=1, l=2, full_output=True)
    # write_fits(testdir3+"/X_"+test_ID, res)

    # PCA
    # res = []
    # for k in range (1,5):
    #       res.append(pca(cube, angles, ncomp=k, cube_ref=ref))
    # write_fits(testdir2+"/X_"+test_ID, np.array(res))


    # ItPCA 1
    # res = pca_it(cube, angles, thr=0, r_out=None, mask_val=0, edge_blend='interp', interp_zeros=True, ker=1, regul=True)
    # write_fits(testdir1+"/X_"+test_ID, res)
    
    # ItPCA 2
    # res = pca_it(cube, angles, thr='auto', r_out=70, mask_val=0, edge_blend='interp', interp_zeros=True, ker=1, regul=True)
    # write_fits(testdir3+"/X_"+test_ID, res)
   
    # RDI-ItPCA vip
    # res = []
    # for k in range (1,2):
    #     res.append(pca_it(cube, angles, cube_ref=ref, strategy="RDI", ncomp=k))
    # write_fits(testdir3+"/X_"+test_ID, np.array(res))

# Write results
#write_fits(dir+"GreeDS_estimation_"+str(r)+"_"+str(l)+"_"+str(r_start), res)

