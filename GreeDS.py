#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 08:43:22 2022

    ______________________________
                GreeDS 
    ______________________________


@author: sand-jrd
"""
import torch
import numpy as np

from kornia import get_rotation_matrix2d, warp_affine


def circle(shape: tuple, r: float, offset=(0.5, 0.5)):
    """ Create circle of 0 in a 2D matrix of ones"
       
       Parameters
       ----------

       shape : tuple
           shape x,y of the matrix
       
       r : float
           radius of the circle
       offset : (optional) float
           offset from the center
       
       Returns
       -------
       M : ndarray
           Zeros matrix with a circle filled with ones
       
    """
    assert(len(shape) == 2 or len(shape) == 3)
    if isinstance(offset, (int, float)): offset = (offset, offset)

    nb_f  = shape[0]  if len(shape) == 3 else 0
    shape = shape[1:] if len(shape) == 3 else shape

    M = np.zeros(shape)
    w, l = shape
    for x in range(0, w):
        for y in range(0, l):
            if pow(x - (w // 2) + offset[0], 2) + pow(y - (l // 2) + offset[1], 2) < pow(r, 2):
                M[x, y] = 1

    if nb_f: M = np.tile(M, (nb_f, 1, 1))

    return 1-M

def GreeDS(cube,angles,r=1,l=10, pup=6, full_output=False):
    """

    Parameters
    ----------
    cube : numpy array
        3D cube of data. shape : (nb_frame, length, width)
    angles : numpy array
        1D array of PA angle. Must be the same length as cube nb_frame
    r : int
        Number of rank to iterate over. The default is 1.
    l : int
        Number of iteration per rank. The default is 10.
    pup : int
        Raduis of the pupil mask
    full_output : bool
        Return the estimation at each iteration

    Returns
    -------
    iter_frames
        Estimated circumstellar per iterations.

    """
    
    # Shapes
    shape = cube.shape[-2:]
    len_img = shape[0]
    nb_frame = len(angles)
    
    # Convert to use torch
    cube = torch.from_numpy(cube)
    # write_fits("cube", cube_dero.numpy())
    angles = torch.from_numpy(angles)

    # Init variables
    if full_output : iter_frames = torch.zeros((l*r,)+shape)
    x_k = torch.zeros(shape)
    
    center: torch.tensor = torch.ones(1, 2)
    center[..., 0] = shape[0]//2 # x
    center[..., 1] = shape[0]//2 # y
    scale: torch.tensor = torch.ones(1,2)
    t, = angles.shape
    pa_derotate_matrix = torch.zeros((nb_frame,2, 3))
    pa_rotate_matrix = torch.zeros((nb_frame,2, 3))

    for i in range(t):
        theta = torch.ones(1) * (angles[i])
        pa_rotate_matrix[i,:,:] = get_rotation_matrix2d(center, theta, scale)
        theta = - torch.ones(1) * (angles[i])
        pa_derotate_matrix[i,:,:] = get_rotation_matrix2d(center, theta, scale)
    
    def rotate_cube(cube) :
        torch_cube = cube.expand(nb_frame,len_img,len_img).unsqueeze(1).float()
        return warp_affine(torch_cube, pa_rotate_matrix, dsize=shape).squeeze(1).detach()
     
    def derotate_cube(cube) :
        torch_cube = cube.expand(nb_frame,len_img,len_img).unsqueeze(1).float()
        return warp_affine(torch_cube, pa_derotate_matrix, dsize=shape).squeeze(1).detach()

    # One iteration of greeDS
    def GreeDS_iter(x,q):
        
         
         R = cube - rotate_cube(x.expand(nb_frame,len_img,len_img))
         U,Sigma,V = torch.pca_lowrank(R.view(nb_frame,len_img*len_img),q=q,niter=4,center=False)
         L = (U @ torch.diag(Sigma) @ V.T).reshape(nb_frame,len_img,len_img)
         
         S = cube - L
         S_der = derotate_cube(S)

         frame = torch.mean(S_der,axis=0)*circle(shape, pup)
         frame *= frame>0
         
         return frame, L
    
    ## Main loop over N_comp and nb_rank.
    
    for ncomp in range(1,r+1):
        
        for ii in range(1,l+1):
            
            x_k1, xl = GreeDS_iter(x_k,ncomp)
            x_k = x_k1.clone()
            
            if full_output : iter_frames[(ncomp-1)*l+(ii-1),:,:] = x_k1
            
    if full_output : return iter_frames.numpy()
    else : return x_k.numpy()

# %% EXEMPLE

from vip_hci.fits import open_fits, write_fits

cube = open_fits("../DISK/HIP67497/cube_crop.fits")
angles = open_fits("../DISK/HIP67497/angles.fits")

res = GreeDS(cube,angles,10,20, full_output=True)

write_fits("res", res)
