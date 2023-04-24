# GreeDS

`I-PCA` (Iterative Principal Component Analysis) tools developed to process ADI cube


Refactored implementation of the original code from [Pairet et al](https://arxiv.org/pdf/2008.05170.pdf)

Updtates to `MAYONNAISE` version :
  - Can be used without the optimization part (`MAYONNAISE`).
  - Option r_start
  - No Deprecated packages (rotation without Kornia)
  - full_outputs options

Version without deprecated dependency


## Install package

Clone the project

```bash
  git clone https://github.com/Sand-jrd/mustard
```

Go to the project directory

```bash
  cd GreeDS
```

Install dependencies

```bash
  py setup.py install
```

## Usage/Exemple

This package contain only one function. All necessay information are in the function comments.
Follow instruction in the [demo](demo.py) to test the algorithm with your own datasets.

Import the function.

```python
  from GreeDS import GreeDS
```

Load your dataset call the function.

```python
  from vip_hci.fits import open_fits
  cube = open_fits("your_cube.fits")
  angles = open_fits("your_PA_angles.fits")
```

Set parameters

```python
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
```

That's it. Call `GreeDS` and get your results

```python
    res = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size, full_output=full_output)
```

## Related

Also check out other package for Exoplanet/disk direct imaging

- [MUSTARD](https://github.com/Sand-jrd/mustard)
Inverse problem approch to process ADI cube

- [Slides](https://docs.google.com/presentation/d/1aPjWJUztfjROtt8BPi8uh6X-vBD5dc81wQ1MhMGGOas/edit) 
Doc about the maths behind the algorithms (GreeDS/MUSTARD) and comparison

- [VIP - Vortex Image Processing package](https://github.com/vortex-exoplanet/VIP)
Tools for high-contrast imaging of exoplanets and circumstellar disks.


## Feedback/Support

You can contact me by email : sjuillard@uliege.be