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
import photutils
import matplotlib.pyplot as plt


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


def cube_rotate(cube, angles, fft=False):
    new_cube = torch.zeros(cube.shape)
    if not fft:
        for ii in range(len(angles)):
            new_cube[ii] = rotate(torch.unsqueeze(cube[ii], 0), -float(angles[ii]),
                                  InterpolationMode.BILINEAR)[0]
        return new_cube
    else:
        for ii in range(len(angles)):
            new_cube[ii] = tensor_rotate_fft(torch.unsqueeze(cube[ii], 0), -float(angles[ii]))
        return new_cube


class GtolNotReached(Exception):
    """Considere increasing gtol or chosing another set of parameters"""
    pass


def find_optimal_iter(res, noise_lim=30, singal_lim=(10, 30), apps=None, gtol=1e-2, win=2, plot=False, saveplot=False,
                      returnSNR=False, app_size=8, l=10, r_start=1, r=10):
    """Find the optimal iteration in two steps : 1-Ensuring signal have converged 2-Minimizing SNR
    
    Parameters
    ----------

    r_start
    r
    l
    saveplot
    app_size
    returnSNR
    plot
    apps
    res : numpy array
        3D cube of GreeDS estimate. shape : (nb_frame, length, width)
        
    noise_lim : int [default=30]
        Limit raduis of noise region
        
    singal_lim : tuple [default=(10,30)] or "app"
        Inner and outter raduis of signal region
        
    gtol : int [gtol=0.1]
        Gradient tolerance
        
    win : int [default=3]
        Moving average window 
        

    Returns
    -------

    res[indx] : numpy array
        Optimal frame estimate
    
    indx : int
        Optimal index frame
  
    """
    size = res.shape[1]
    pup = circle((size, size), 8) - circle((size, size), size // 2)

    ## Defining Noise and signal region & Computing flx variation

    if str(noise_lim) == "app":
        img = res[3]
        plt.title("Click to define apperture of noise region")
        plt.imshow(pup * img, vmax=np.percentile(img, 99))
        apps = plt.ginput(n=1)[0]
        siftx = apps[0]
        sifty = apps[1]

        fwhm_aper = photutils.CircularAperture([siftx, sifty], app_size)
        noise = fwhm_aper.to_mask().to_image(img.shape)
        flx_noise = np.array(
            [photutils.aperture_photometry(frame, fwhm_aper, method='exact')["aperture_sum"] for frame in
             res]).flatten()
        plt.close("all")
    else:
        noise = circle((size, size), size // 2) - circle((size, size), size // 2 - noise_lim)
        flx_noise = np.sum(res * noise, axis=(1, 2)) / np.sum(noise)

    if str(singal_lim) == "app":
        img = res[10]
        plt.figure("Click to define apperture of signal region")
        plt.title("Click to define apperture of signal region")
        plt.imshow(pup * img, vmax=np.percentile(img, 99.99))
        if apps is None: apps = plt.ginput(n=1)[0]
        print(apps)
        siftx = apps[0]
        sifty = apps[1]
        fwhm_aper = photutils.CircularAperture([siftx, sifty], app_size)
        signal = fwhm_aper.to_mask().to_image(img.shape)
        flx_sig = np.array([photutils.aperture_photometry(frame, fwhm_aper, method='exact')["aperture_sum"] for frame in
                            res]).flatten()
        plt.close("Click to define apperture of signal region")

    else:
        signal = circle((size, size), singal_lim[1]) - circle((size, size), singal_lim[0])
        flx_sig = np.sum(res * signal, axis=(1, 2)) / np.sum(signal)

    # Computing gradient to find the convergence of signal
    grad = (flx_sig[0:-1] - flx_sig[1:]) / np.mean(flx_sig)
    if win: grad = np.convolve(grad, np.ones(win), 'valid') / win  # moving avg grads

    valid_conv = np.flatnonzero(np.convolve(abs(grad) < gtol, np.ones(win, dtype=int)) == win)
    if len(valid_conv) < 1:
        while len(valid_conv) < 1:
            gtol *= 2
            print("gtol too small, increasing tolerernce :  {:2e}".format(gtol))
            valid_conv = np.flatnonzero(np.convolve(abs(grad) < gtol, np.ones(win, dtype=int)) == win)
            if gtol > 1: valid_conv = [len(grad) + 2 - win - 1]

    conv_indx = valid_conv[0] - 2 + win

    SNR = flx_sig / flx_noise
    indx = np.argmax(SNR[conv_indx:]) + conv_indx

    minortick = np.array(range(len(res)))
    if l == "incr":
        majortick = []
        tmp = 0
        for k in range(0, r - r_start):
            majortick.append(tmp + k)
            tmp = tmp + k
        majortick = np.array(majortick)
    else:
        majortick = np.array(range(0, r - r_start)) * l

    majroticklab = ["rank " + str(k) for k in range(r_start, r)]

    if plot or saveplot:
        if not plot: plt.ioff()
        plt.close("Find Optimal Iteration")
        plt.figure("Find Optimal Iteration", (16, 9))
        point_param = {'color': "black", 'markersize': 7, 'marker': "o"}
        text_param = {'color': "black", 'weight': "bold", 'size': 10, 'xytext': (-10, 20),
                      'textcoords': 'offset points'}
        plt.subplot(3, 2, 1), plt.imshow(noise), plt.title("Noise")
        plt.subplot(3, 2, 2), plt.imshow(signal), plt.title("Signal")

        ax = plt.subplot(3, 2, 3)
        plt.plot(flx_sig / np.mean(flx_sig), label="Variation gradient")
        plt.plot([conv_indx], [flx_sig[conv_indx] / np.mean(flx_sig)], **point_param)
        plt.annotate('Convergence', xy=(indx, grad[conv_indx]), **text_param)
        plt.legend(loc="lower right")
        ax.set_xticks(minortick, minor=True)  # labels=np.array(list(range(1,10,3))*10)
        ax.tick_params(axis='x', which='minor', length=3, width=1, colors="gray", pad=1, labelsize=8)
        ax.tick_params(axis='x', which='major', length=5, width=1, colors='r', labelrotation=30, labelsize=10)
        ax.set_xticks(majortick, labels=majroticklab, minor=False)

        ax = plt.subplot(3, 2, 5)
        plt.plot(grad, label="Variation gradient")
        plt.plot([gtol] * len(grad), color="red", label="tolerance")
        plt.plot([-gtol] * len(grad), color="red")
        plt.plot([conv_indx], [grad[conv_indx]], **point_param)
        plt.annotate('Convergence', xy=(indx, grad[conv_indx]), **text_param)
        plt.legend(loc="lower right")
        ax.set_xticks(minortick, minor=True)  # labels=np.array(list(range(1,10,3))*10)
        ax.tick_params(axis='x', which='minor', length=3, width=1, colors="gray", pad=1, labelsize=8)
        ax.tick_params(axis='x', which='major', length=5, width=1, colors='r', labelrotation=30, labelsize=10)
        ax.set_xticks(majortick, labels=majroticklab, minor=False)

        ax = plt.subplot(3, 2, 4)
        plt.plot(flx_sig / np.mean(flx_sig), color="tab:orange", label="RELATIVE Signal variation")
        plt.plot(flx_noise / np.mean(flx_noise), color="tab:blue", label="RELATIVE Noise variation")
        plt.plot([indx], [flx_sig[indx] / np.mean(flx_sig)], **point_param)
        plt.annotate('Max SNR', xy=(indx, flx_sig[indx] / np.mean(flx_sig)), **text_param)
        plt.legend(loc="lower right")
        ax.set_xticks(minortick, minor=True)  # labels=np.array(list(range(1,10,3))*10)
        ax.tick_params(axis='x', which='minor', length=3, width=1, colors="gray", pad=1, labelsize=8)
        ax.tick_params(axis='x', which='major', length=5, width=1, colors='r', labelrotation=30, labelsize=10)
        ax.set_xticks(majortick, labels=majroticklab, minor=False)

        ax = plt.subplot(3, 2, 6)
        plt.plot(SNR, label="SNR"),
        plt.plot([indx], [SNR[indx]], **point_param)
        plt.annotate('Max SNR', xy=(indx, SNR[indx]), **text_param)
        plt.legend(loc="lower right")
        ax.set_xticks(minortick, minor=True)  # labels=np.array(list(range(1,10,3))*10)
        ax.tick_params(axis='x', which='minor', length=3, width=1, colors="gray", pad=1, labelsize=8)
        ax.tick_params(axis='x', which='major', length=5, width=1, colors='r', labelrotation=30, labelsize=10)
        ax.set_xticks(majortick, labels=majroticklab, minor=False)

        if saveplot: plt.savefig(saveplot + ".png")
        if not plot: plt.close("all")

    if returnSNR:
        return res[indx], indx, SNR[indx], np.mean(grad[conv_indx:])
    else:
        return res[indx], indx


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


# %%

def find_param(cube, angles, refs=None, savedir="./"):
    r = 20
    maxSNR = 0
    opti_set = "1_1_1"
    for r_start in range(1, 5):
        for l in range(1, 10):
            res_l = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=6, refs=refs, x_start=None, full_output=1,
                           returnL=False)
            name = "findOPtim_" + str(r) + "_" + str(r_start) + "_" + str(l)

            try:
                res, indx, SNR, grad = find_optimal_iter(res_l, plot=True, saveplot=savedir + name, returnSNR=True)
                if SNR > maxSNR:
                    opti_set = str(r) + "_" + str(r_start) + "_" + str(l)
                    maxSNR = SNR
            except GtolNotReached:
                pass

            write_fits(savedir + name + "_opti_is_" + str(indx + 1), res_l)

    return opti_set


# %%

def GreeDSRDI(cube, angles, ref, r=1, l=2, pup=6, full_output=0):
    """

    Parameters
    ----------
    ref
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
        
    ref : numpy array
        3D cube of reference frames. shape = (nb_frame, length, width)
        
    full_output : int (0 to  3)
        Choose to return :  
        *  0/False -> only last estimation 
        *  1/True  -> every iter over r*l
        *  2       -> every iter over r
        *  3       -> every iter over l
        

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
    nb_frame_es = ref.shape[0]

    # Convert to use torch
    cube = torch.from_numpy(cube)
    ref = torch.from_numpy(ref)

    angles = torch.from_numpy(angles)
    pup = circle(shape, pup)

    # Init variables
    iter_frames = torch.zeros((l,) + shape)

    x_k = torch.zeros(shape)

    iter_frames = []
    iter_L = []

    # One iteration of greeDS
    def GreeDS_iter(x, ncomp):
        
        U, Sigma, V = torch.pca_lowrank(ref.reshape(nb_frame_es, len_img * len_img), q=ncomp, niter=1, center=False)

        R = cube - cube_rotate(x.expand(nb_frame, len_img, len_img), -angles)
        Ur, Sigmar, Vr = torch.pca_lowrank(R.reshape(nb_frame, len_img * len_img), q=ncomp, niter=1, center=False)
        L = (Ur @ torch.diag(Sigmar) @ V.T).reshape(nb_frame, len_img, len_img)

        L *= L > 0
        S_der = cube_rotate(cube - L, angles)

        frame = torch.mean(S_der, axis=0) * pup
        frame *= frame > 0

        return frame, L
    
    for ncomp in range(1, r + 1):
        for ii in range(1, l + 1):
    
            x_k1, xl = GreeDS_iter(x_k, ncomp)
            x_k = x_k1.clone()
            if full_output: 
                iter_frames.append(x_k1.numpy())
                iter_L.append(xl.numpy())

    if full_output:
        iter_frames = np.array(iter_frames)
        iter_L = np.array(iter_L)
        return iter_frames, iter_L
    else:
        return x_k.numpy(), xl.numpy()


def RDI(cube, angles, ref, r=3, pup=6):
    """

    Parameters
    ----------
    r
    cube : numpy array
        3D cube of data. shape : (nb_frame, length, width)
        
    angles : numpy array
        1D array of PA angle. Must be the same length as cube nb_frame
        
    pup : int
        Raduis of the pupil mask
        
    ref : numpy array
        3D cube of reference frames. shape = (nb_frame, length, width)

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
    nb_frame_es = ref.shape[0]

    # Convert to use torch
    cube = torch.from_numpy(cube)
    ref = torch.from_numpy(ref)

    angles = torch.from_numpy(angles)
    pup = circle(shape, pup)

    torch.zeros(shape)

    U, Sigma, V = torch.pca_lowrank(ref.reshape(nb_frame_es, len_img * len_img), q=r, niter=1, center=False)

    R = cube
    U, Sigma, Vr = torch.pca_lowrank(R.reshape(nb_frame, len_img * len_img), q=r, niter=1, center=False)
    L = (U @ torch.diag(Sigma) @ V.T).reshape(nb_frame, len_img, len_img)

    L *= L > 0
    S_der = cube_rotate(cube - L, angles)

    frame = torch.mean(S_der, axis=0) * pup
    frame *= frame > 0

    return frame.numpy(), L.numpy()

