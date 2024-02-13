#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:00:14 2023

@author: sand-jrd


Test the three algorithms on real datas. 

"""

from GreeDS import GreeDS, GreeDSRDI, find_param
from mustRDI import mustardRDI, theoretical_lim
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


from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

# %%

datadir = "./test_cubes_sphere/"

suf = "H2"
star_name = "MWC - "+suf

cube = open_fits(datadir+"/cube")
ref = open_fits(datadir+"/ref")
angles = open_fits(datadir+"/angles")

ref = cube_crop_frames(ref, cube.shape[-1], force=True)
ref_c = ref[:int(cube.shape[0]*1)]

# %% Param

# For save
comment = "first"
savedir = "./"+star_name+comment+"/"
if not isdir(savedir) : mkdir(savedir)

## -- For algos
pup_size=8

l="incr"
r=10
r_start=1
full_output=False

# %% Processing

res_must, values = mustardRDI(cube, angles, ref, pup_size, save="./")#testdir+"/X_"+test_ID)
write_fits(savedir+"mustRDI",res_must)

res_GreeDS = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output, RDI=ref_c, perfect=True)
write_fits(savedir+"GreeDSRDI",res_must)

res_PCA = pca_it(cube, angles, thr='auto', r_out=70, mask_val=0, edge_blend='interp', interp_zeros=True, ker=1, regul=True)
write_fits(savedir+"ipcaRDI",res_must)

# %% Plots

shape=128
perX = 99.8
perL = 99
perD = 99

xticks_lab = np.array([-0.5, 0, 0.5])
yticks_lab = np.array([-0.5, 0, 0.5])
plsca = 0.01226
xticks = (xticks_lab/plsca) + shape/2
yticks = (yticks_lab/plsca) + shape/2


pup = circle((shape,shape),shape//2) -circle((shape,shape),10)

def genere_args(Tm, M, Dtype, must=False):
    if Dtype == "X": 
        arg = {"cmap":"magma"}
        per = perX
        vmin = np.percentile(M[np.where(M>0)], 10)
        Tm[np.where(Tm<=0)] = vmin
        M[np.where(M<=0)] = vmin
        Tm[np.where(Tm<=0)] = vmin
        arg["norm"]= LogNorm(vmin=vmin, vmax=np.percentile(M, per))
        print(vmin)
    elif Dtype == "L": 
        arg = {"cmap":"jet"}
        per = perL
        arg["vmax"]=np.percentile(Tm, per)
    else :
        arg = {"cmap":"seismic"}
        per = perD
        arg["vmax"]=np.percentile(M, per)
    arg["X"]=Tm
    return arg


fig = plt.figure("real-data " +star_name, figsize=(9.5,8.5))

X_must =  pup*np.sqrt(abs(frame_crop(res_must, shape)))
X_PCA =  pup*np.sqrt(abs(frame_crop(res_PCA, shape)))

best_frame=10
X_GreeDS =  pup*np.sqrt(abs(frame_crop(res_GreeDS[best_frame-1], shape)))


plt.subplot(1,3,1)
heatmap = plt.imshow(**genere_args(X_must, X_must, "X", True))
plt.title("MUSTAR ARDI")
plt.gca().invert_yaxis()
plt.xticks(xticks, labels = xticks_lab)
plt.yticks(yticks, labels = yticks_lab)

plt.subplot(1,3,3)
plt.imshow(**genere_args(X_PCA, X_must, "X", True))
plt.title("I-proj PCA with RDI")
plt.gca().invert_yaxis()
plt.xticks(xticks, labels = xticks_lab)
plt.yticks(yticks, labels = yticks_lab)

plt.subplot(1,3,2)
plt.imshow(**genere_args(X_GreeDS, X_must, "X", True))
plt.title("I-PCA with ARDI")
plt.gca().invert_yaxis()
plt.xticks(xticks, labels = xticks_lab)
plt.yticks(yticks, labels = yticks_lab)


fig.subplots_adjust(right=0.8)
im_ratio = 4/5
cbar = fig.colorbar(heatmap, ax=[plt.subplot(1,3,1), plt.subplot(1,3,2), plt.subplot(1,3,3)], fraction=0.046*im_ratio, pad=0.04, shrink=0.9)
