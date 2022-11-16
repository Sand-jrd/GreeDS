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

from GreeDS import GreeDS
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_crop_frames

## Load data

dir = "your_directory"
cube = open_fits(dir+"your_cube.fits")#[my_channel] # Must be one channel cube 
angles = open_fits(dir+"your_PA_angles.fits")

## Set parameters

r = 20  # Iteration over PCA-rank
l = 20  # Iteration per rank
r_start = 1 # PCA-rank to start iteration (good for faint signal)
pup_size = 6 # Raduis of numerical mask to hide coro

full_output = 3 # Return estimation at each iter (needed to search opti params) 
# If 0 -> only last estimation 
# if 1 -> every iter over r*l
# if 2 -> every iter over r
# if 3 -> every iter over l

# Crop you cube (optional)
crop_size = 200
cube = cube_crop_frames(cube, crop_size)

# Greeds
res = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output)

# Write results
write_fits(dir+"GreeDS_estimation_"+str(r)+"_"+str(l)+"_"+str(r_start), res)
