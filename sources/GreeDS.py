#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 08:43:22 2022

    ______________________________
                GreeDS 
    ______________________________

GreeDS algorithm from Pairet etal 2020.

Basic implemented that works independently from MAYONNAISE.

Require the dependancy torch and kornia

@author: sand-jrd
"""

import torch
import numpy as np
from rotation import tensor_rotate_fft
from torchvision.transforms.functional import (rotate, InterpolationMode)

def cube_rotate(cube, angles, fft=False):
    new_cube = torch.zeros(cube.shape)
    if not fft:
        for ii in range(len(angles)):
            new_cube[ii] = rotate(torch.unsqueeze(cube[ii], 0), -float(angles[ii]),
                                  InterpolationMode.BILINEAR)[0]
        return new_cube
    else:
        for ii in range(len(angles)):
            new_cube[ii] = tensor_rotate_fft((torch.unsqueeze(cube[ii], 0), -float(angles[ii]))
        return new_cube


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
    assert (len(shape) == 2 or len(shape) == 3)
    if isinstance(offset, (int, float)): offset = (offset, offset)

    nb_f = shape[0] if len(shape) == 3 else 0
    shape = shape[1:] if len(shape) == 3 else shape

    M = np.zeros(shape)
    w, l = shape
    for x in range(0, w):
        for y in range(0, l):
            if pow(x - (w // 2) + offset[0], 2) + pow(y - (l // 2) + offset[1], 2) < pow(r, 2):
                M[x, y] = 1

    if nb_f: M = np.tile(M, (nb_f, 1, 1))

    return 1 - M


def GreeDS(cube, angles, r=1, l=10, r_start=1, pup=6, refs=None, x_start=None, full_output=0, returnL=False,
           returntype="numpy"):
    """

    Parameters
    ----------
    x_start
    cube : numpy array
        3D cube of data. shape : (nb_frame, length, width)
        
    angles : numpy array
        1D array of PA angle. Must be the same length as cube nb_frame
        
    r : int
        Number of rank to iterate over. The default is 1.
        
    l : int or str {'incr'}
        Number of iteration per rank. The default is 10.
        If set to 'incr', the number of iteration per rank will increase with rank.

    r_start : int
        First rank estimate, r_start < r
        GreeDS will iterate from rank r-start to r 
        
    pup : int
        Raduis of the pupil mask
        
    refs : numpy array
        3D cube of reference frames. shape = (nb_frame, length, width)

    returntype : {"numpy", "tensor"}
        Type of the function output

    full_output : int (0 to  3)
        Choose to return :  
        *  0/False -> only last estimation 
        *  1/True  -> every iter over r*l
        *  2       -> every iter over r
        *  3       -> every iter over l
        
    returnL : bool
        Return PSF estimation
        
    Returns
    -------
    x_k [full_ouputs=False]
        Estimated circumstellar signal. 
        
    iter_frames [full_ouputs=True]
        Estimated circumstellar signal x_k for different iterations.

    """

    # Shapes
    shape = cube.shape[-2:]
    len_img = shape[0]
    nb_frame = len(angles)
    nb_frame_es = len(angles)

    # References
    if refs is not None:
        assert (refs.shape[-2:] == shape)
        refs = torch.from_numpy(refs)
        print("Cube filled with " + str(int(100 * refs.shape[0] / nb_frame)) + " percent of reference frames")
        nb_frame_es = len(angles) + refs.shape[0]

    # Convert to use torch
    cube = torch.from_numpy(cube)

    angles = torch.from_numpy(angles)
    pup = 1 if pup == 0 else circle(shape, pup)

    iter_frames = []
    iter_L = []

    x_k = torch.zeros(shape)
    if x_start is not None: x_k = torch.from_numpy(x_start)

    incr = True if l == "incr" else False

    # One iteration of greeDS
    def GreeDS_iter(x, q):

        R = cube - cube_rotate(x.expand(nb_frame, len_img, len_img), -angles)

        if refs is not None:
            R = torch.cat((R, refs))

        U, Sigma, V = torch.pca_lowrank(R.view(nb_frame_es, len_img * len_img), q=q, niter=1, center=False)
        L = (U @ torch.diag(Sigma) @ V.T).reshape(nb_frame_es, len_img, len_img)

        if refs is not None: L = L[:nb_frame]
        L *= L > 0

        S_der = cube_rotate(cube - L, angles)

        frame = torch.mean(S_der, axis=0) * pup
        frame *= frame > 0

        return frame, L

    ## Main loop over N_comp and nb_rank.
    for ncomp in range(r_start, r + 1):

        if incr: l = ncomp - r_start + 1

        for _ in range(1, l + 1):

            x_k1, xl = GreeDS_iter(x_k, ncomp)
            x_k = x_k1.clone()

            if full_output == 1:
                iter_frames.append(x_k1.numpy())
                if returnL: iter_L.append(xl.numpy())
            if full_output == 3 and ncomp == r + 1: iter_frames.append(x_k1.numpy())

        if full_output == 2: iter_frames.append(x_k1.numpy())

    iter_frames = np.array(iter_frames)
    iter_L = np.array(iter_L)
    if returntype == "numpy":
        x_k = x_k.numpy()
        xl = xl.numpy()
    if returnL:
        if full_output:
            return iter_frames, iter_L
        else:
            return x_k, xl

    if full_output:
        return iter_frames
    else:
        return x_k
