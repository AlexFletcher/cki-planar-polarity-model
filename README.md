# CKIε planar polarity model

Code to reproduce the simulation results in:

> Hossein SR\*, Shao H\*, Warrington SJ\*, Strutt H, Fletcher AG, Strutt D (2026). Polarised recruitment and diffusion of Casein Kinase Iε couples local protein clustering to cell-scale planar polarity in the *Drosophila* wing. *In submission.*

The script `regenerate_figures.py` solves the two reaction-diffusion models described in the STAR Methods, the mutual destabilisation model and the CKIε model. The models are solved on a periodic one-dimensional domain representing the perimeter of a cell facing a neighbour. The script generates every simulation-derived panel in the paper.

## Requirements

Python 3.12 or later, with NumPy, SciPy and Matplotlib:

```
pip install numpy scipy matplotlib
```

No other data are needed. The FRAP values used to constrain the model are written into the script.

## Reproducing the figures

All figures in the paper use a grid of 800 points (`--N 800`) and are reported at a nondimensional time of 2 × 10⁴, about 18 h. The default grid is 200 points, which is for quick checks only.

| Figure | Command |
|---|---|
| 2H | `--panels 2H` |
| 2I–L | `--panels 2IL` |
| 6B–D | `--panels 6BD` |
| 6E–M | `--panels 6EM` |
| 6R,S | `--panels 6NO` (labelled 6N,O in the script's output) |
| S3C–J | `--sensitivity MD` |
| S6A–G | `--sensitivity CKI` |
| S6H | `--sigma-test` |
| S6I,J | `--pk-dose` |
| S7A | `--time-course` |
| S7B,C | `--mesh-test --mesh-N 800 1600` |
| S7D | `--panels width` |
| S7E | `--width-sigma 0.005 0.01 0.02` |

For example, to regenerate all main-text panels with 35 simulations per condition on 8 cores:

```
python regenerate_figures.py --panels all --N 800 --reps 35 --jobs 8
```

A two-minute check that everything runs:

```
python regenerate_figures.py --smoke
```

`python regenerate_figures.py --help` lists all options. The docstring at the top of the script describes the parameterisation, initial conditions and the main caveats.

## Outputs and caching

Figures and cached results are written to `figures/` (change this with `--outdir`). Results are cached as `.npz` files tagged by grid size, number of simulations, end time and a hash of the parameter set. A rerun with the same settings redraws the figures without recomputing. Use `--force` to recompute. Runs that change a parameter from its default get a suffix on their figure filenames, so they never overwrite the production figures.

## Run time

One simulation of the CKIε model takes about 50 s at 200 grid points and several minutes at 800. Figure 6R,S needs 420 simulations, so use `--jobs` to run in parallel. Memory use grows with the number of jobs and the grid size. At 1600 grid points, use `--jobs 2` or `3` on a machine with 16 GB of memory.

## Licence

This code is released under the BSD 3-Clause licence; see `LICENSE`.

## Citation

If you use this code, please cite the paper above. 

## Funding

This work was supported by an EPSRC award to Alexander Fletcher and David Strutt (EP/W024144/1), and a Wellcome Trust Senior Fellowship (210630/Z/18/Z) and Discovery Award (305132/Z/23/Z) to David Strutt. 

## Contact

Alexander Fletcher, School of Mathematical and Physical Sciences, University of Sheffield.
