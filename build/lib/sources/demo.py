"""
Created on Thu Nov 10 08:43:22 2022

    ______________________________
                GreeDS - DEMO
    ______________________________

GreeDS algorithm from Pairet etal 2020.
Basic implemented that works independently from MAYONNAISE.
Require the dependancy torch and kornia

@author: sand-jrd
"""

from GreeDS import GreeDS
from vip_hci.fits import open_fits, write_fits

cube = open_fits("../../DISK/HIP67497/cube_crop.fits")
angles = open_fits("../../DISK/HIP67497/angles.fits")

r = 10  # Iteration over PCA-rank
l = 10  # Iteration per rank
full_output = False  # If True, return every iterations. Better if you are searching optimized param

res = GreeDS(cube, angles, r=r, l=l, full_output=2)
write_fits("GreeDS_estimation", res)
