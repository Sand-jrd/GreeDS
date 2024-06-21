import numpy as np
import torch
import torch.fft as tf

def frame_center(array, verbose=False):
    """
    Return the coordinates y,x of the frame(s) center.
    If odd: dim/2-0.5
    If even: dim/2

    Parameters
    ----------
    array : 2d/3d/4d numpy ndarray
        Frame or cube.
    verbose : bool optional
        If True the center coordinates are printed out.

    Returns
    -------
    cy, cx : int
        Coordinates of the center.

    """
    if array.ndim == 2:
        shape = array.shape
    elif array.ndim == 3:
        shape = array[0].shape
    elif array.ndim == 4:
        shape = array[0, 0].shape
    else:
        raise ValueError('`array` is not a 2d, 3d or 4d array')

    cy = shape[0] / 2
    cx = shape[1] / 2

    if shape[0] % 2:
        cy -= 0.5
    if shape[1] % 2:
        cx -= 0.5

    if verbose:
        print('Center px coordinates at x,y = ({}, {})'.format(cx, cy))

    return int(cy), int(cx)


def tensor_rotate_fft(tensor: torch.Tensor, angle: float) -> torch.Tensor:
    """ Rotates Tensor using Fourier transform phases:
        Rotation = 3 consecutive lin. shears = 3 consecutive FFT phase shifts
        See details in Larkin et al. (1997) and Hagelberg et al. (2016).
        Note: this is significantly slower than interpolation methods
        (e.g. opencv/lanczos4 or ndimage), but preserves the flux better
        (by construction it preserves the total power). It is more prone to
        large-scale Gibbs artefacts, so make sure no sharp edge is present in
        the image to be rotated.
        /!\ This is a blindly coded adaptation for Tensor of the vip function rotate_fft
        (https://github.com/vortex-exoplanet/VIP/blob/51e1d734dcdbee1fbd0175aa3d0ab62eec83d5fa/vip_hci/preproc/derotation.py#L507)
        /!\ This suppose the frame is perfectly centred
        ! Warning: if input frame has even dimensions, the center of rotation
        will NOT be between the 4 central pixels, instead it will be on the top
        right of those 4 pixels. Make sure your images are centered with
        respect to that pixel before rotation.
    Parameters
    ----------
    tensor : torch.Tensor
        Input image, 2d array.
    angle : float
        Rotation angle.
    Returns
    -------
    array_out : torch.Tensor
        Resulting frame.
    """
    y_ori, x_ori = tensor.shape[1:]

    while angle < 0:
        angle += 360
    while angle > 360:
        angle -= 360

    if angle > 45:
        dangle = angle % 90
        if dangle > 45:
            dangle = -(90 - dangle)
        nangle = int(np.rint(angle / 90))
        tensor_in = torch.rot90(tensor, nangle, [1, 2])
    else:
        dangle = angle
        tensor_in = tensor.clone()

    if y_ori % 2 or x_ori % 2:
        # NO NEED TO SHIFT BY 0.5px: FFT assumes rot. center on cx+0.5, cy+0.5!
        tensor_in = tensor_in[:, :-1, :-1]

    a = np.tan(np.deg2rad(dangle) / 2).item()
    b = -np.sin(np.deg2rad(dangle)).item()

    y_new, x_new = tensor_in.shape[1:]
    arr_xy = torch.from_numpy(np.mgrid[0:y_new, 0:x_new])
    cy, cx = frame_center(tensor[0])
    arr_y = arr_xy[0] - cy
    arr_x = arr_xy[1] - cx

    s_x = tensor_fft_shear(tensor_in, arr_x, a, ax=2)
    s_xy = tensor_fft_shear(s_x, arr_y, b, ax=1)
    s_xyx = tensor_fft_shear(s_xy, arr_x, a, ax=2)

    if y_ori % 2 or x_ori % 2:
        # set it back to original dimensions
        array_out = torch.zeros([1, s_xyx.shape[1]+1, s_xyx.shape[2]+1])
        array_out[0, :-1, :-1] = torch.real(s_xyx)
    else:
        array_out = torch.real(s_xyx)

    return array_out


def tensor_fft_shear(arr, arr_ori, c, ax):
    ax2 = 1 - (ax-1) % 2
    freqs = tf.fftfreq(arr_ori.shape[ax2], dtype=torch.float64)
    sh_freqs = tf.fftshift(freqs)
    arr_u = torch.tile(sh_freqs, (arr_ori.shape[ax-1], 1))
    if ax == 2:
        arr_u = torch.transpose(arr_u, 0, 1)
    s_x = tf.fftshift(arr)
    s_x = tf.fft(s_x, dim=ax)
    s_x = tf.fftshift(s_x)
    s_x = torch.exp(-2j * torch.pi * c * arr_u * arr_ori) * s_x
    s_x = tf.fftshift(s_x)
    s_x = tf.ifft(s_x, dim=ax)
    s_x = tf.fftshift(s_x)

    return s_x
