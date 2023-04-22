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

from GreeDS import GreeDS
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_crop_frames

# %%  Load data

dir = "your_directory"
cube = open_fits(dir+"your_cube.fits")#[my_channel] # Must be one channel cube 
angles = open_fits(dir+"your_PA_angles.fits")

# (optional) Reference frames. Add a data-driven prior using referene frames
ref = open_fits(dir+"your_cube_ref.fits") or None

# %% Set parameters

r = 20  # Iteration over PCA-rank
l = 20  # Iteration per rank
r_start  = 1 # PCA-rank to start iteration (good for faint signal)
pup_size = 6 # Raduis of numerical mask to hide coro

# Outputs (default 1) 
full_output = 3 
#  0/False -> only last estimation 
#  1/True  -> every iter over r*l
#  2       -> every iter over r
#  3       -> every iter over l

# (optional) Crop you cube 
crop_size = 200
cube = cube_crop_frames(cube, crop_size)

# %% Greeds
res = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output, RDI=ref)

# Write results
write_fits(dir+"GreeDS_estimation_"+str(r)+"_"+str(l)+"_"+str(r_start), res)
