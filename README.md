# Derivative-informed neural operators for optimization under uncertainty 
The this repository contains code for the numerical examples in the paper "Efficient PDE-constrained optimization under high dimensional uncertainty using derivative-informed neural operators". 
Specifically, it includes the data generation, training, and OUU codes for the semilinear elliptic PDE problem and the 2D Navier--Stokes flow control problem. Data generation code for 3D example is not included but can be made available upon request (email the author at `dc.luo@utexas.edu`).

## Installation 
The code uses `FEniCS` for finite element computations and `tensorflow` for machine learning. This along with additional requirements are listed in `environment.yml`, and can be installed via `conda` using 

```conda env create -f environment.yml```

Additionally, the code makes use of `hIPPYLib`, `hIPPYflow` and `SOUPy` from the `hIPPYLib` organization to handle the data generation. `SOUPy` is also used for optimization under uncertainty. We suggest cloning these repositories 

```
git clone https://github.com/hippylib/hippylib.git
git clone https://github.com/hippylib/hippyflow.git
git clone https://github.com/hippylib/soupy.git
```

and setting the path to their base directories
```
conda activate mr_dino
conda env config vars set HIPPYLIB_PATH=path/to/hippylib
conda env config vars set HIPPYFLOW_PATH=path/to/hippyflow
conda env config vars set SOUPY_PATH=path/to/soupy
conda deactivate
```