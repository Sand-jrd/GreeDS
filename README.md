# GreeDS

`I-PCA` (Iterative Principal Component Analysis) tools developed to process ADI cube.

Refactored implementation of the [original code](https://github.com/bpairet/mayo_hci) from [Pairet et al](https://arxiv.org/pdf/2008.05170.pdf).
Can be used independently of the optimization Inverse Problem part (`MAYONNAISE`) + few fixes such as deprecated packages

Updates :
  - More options such as: choosing starting rank, having an incremental number of iterations per rank, and also output options.
  - **NEW**: Can be used with references thus leveraging Angular and Reference Differential Imaging Strategy (ARDI).

## Install package

Clone the project:

```bash
git clone https://github.com/Sand-jrd/mustard
```

Go to the project directory:

```bash
cd GreeDS
```

Install dependencies:

```bash
py ./sources/setup.py install
```

## Usage/Example

This package contains only one function. All necessary information is in the function comments. Follow instructions in the [demo](demo.py) or [notebooek](demo.ipynb) to test the algorithm with your own datasets.

Import the function:

```python
from GreeDS import GreeDS
```

Load your dataset and call the function:

```python
from vip_hci.fits import open_fits
cube = open_fits("your_cube.fits")
angles = open_fits("your_PA_angles.fits")

# Optional
ref = open_fits("your_refs.fits")

```

Set parameters:

```python
r = 10  # Iteration over PCA-rank
l = 10  # Iteration per rank
r_start = 1  # PCA-rank to start iteration (good for faint signals)
pup_size = 3  # Radius of numerical mask to hide coro
```

Call `GreeDS` and get your results:

```python
res = GreeDS(cube, angles, r=r, l=l, r_start=r_start, pup=pup_size)
```

## Related

Also check out other packages for Exoplanet/disk direct imaging:

- [MUSTARD](https://github.com/Sand-jrd/mustard): Inverse problem approach to process ADI cube.
- [VIP - Vortex Image Processing package](https://github.com/vortex-exoplanet/VIP): Tools for high-contrast imaging of exoplanets and circumstellar disks.

Also see docs about the maths behind the algorithms (GreeDS/MUSTARD) and their comparison:

- [Inverse-problem versus principal component analysis methods for angular differential imaging of circumstellar disks. The mustard algorithm](https://ui.adsabs.harvard.edu/abs/2023A%26A...679A..52J/abstract)
- [Slides](https://docs.google.com/presentation/d/1aPjWJUztfjROtt8BPi8uh6X-vBD5dc81wQ1MhMGGOas/edit)

## Feedback/Support

You can contact me by email: sjuillard@uliege.be
```
