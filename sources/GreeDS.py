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
from torchvision.transforms.functional import rotate
from torchvision.transforms.functional import InterpolationMode


def cube_rotate(cube, angles, mode="fft"):
    new_cube = torch.zeros(cube.shape)
    for ii in range(len(angles)):
        new_cube[ii] = rotate(torch.unsqueeze(cube[ii], 0),float(angles[ii]),
                                  InterpolationMode.BILINEAR)[0]
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


def GreeDS(cube, angles, r=1, l=10, r_start=1, pup=6, full_output=0):
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
    angles = torch.from_numpy(angles)

    # Init variables
    if full_output == 1 : iter_frames = torch.zeros((l * (r-r_start+1),) + shape)
    elif full_output == 2 : iter_frames = torch.zeros((r,) + shape)
    elif full_output == 3 : iter_frames = torch.zeros((l,) + shape)

    x_k = torch.zeros(shape)

    # One iteration of greeDS
    def GreeDS_iter(x, q):

        R = cube - cube_rotate(x.expand(nb_frame, len_img, len_img), angles)

        U, Sigma, V = torch.pca_lowrank(R.view(nb_frame, len_img * len_img), q=q, niter=4, center=False)
        L = (U @ torch.diag(Sigma) @ V.T).reshape(nb_frame, len_img, len_img)

        S_der = cube_rotate(cube - L, -angles)

        frame = torch.mean(S_der, axis=0) * circle(shape, pup)
        frame *= frame > 0

        return frame, L

    ## Main loop over N_comp and nb_rank.
    for ncomp in range(1, r + 1):

        for ii in range(1, l + 1):

            x_k1, xl = GreeDS_iter(x_k, ncomp)
            x_k = x_k1.clone()

            if full_output == 1 : iter_frames[(ncomp - r_start) * l + (ii - 1), :, :] = x_k1
            if full_output == 3 : iter_frames[ii - 1, :, :] = x_k1

        if full_output == 2 : iter_frames[ncomp - r_start, :, :] = x_k1


    if full_output:
        return iter_frames.numpy()
    else:
        return x_k.numpy()
