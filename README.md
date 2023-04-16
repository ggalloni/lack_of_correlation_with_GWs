# lack_of_correlation_with_GWs

This repository collects the codes to run the analysis on the lack-of-correlation anomaly using Gravitational Waves (GWs). 

## Dependencies

These are the necessary dependencies to run the analysis:
* pathlib
* ray
* numba
* numpy
* pandas
* scipy
* healpy
* pymaster
* progressbar
* time
* matplotlib

## Repeating the analysis

To perform the analysis it is sufficient to run `complete_pipeline.py`, given that you have all the necessary dependencies. Alternatively, if you want to run the analysis on a single $\ell_{\rm max}$ choice, run one of `run_lmax4.py`, `run_lmax6.py` or `run_lmax10.py`.
