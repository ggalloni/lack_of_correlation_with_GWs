# lack_of_correlation_with_GWs

This repository collects the codes to run the analysis on the lack-of-correlation anomaly using Gravitational Waves (GWs). 

## Necessary data

Note that you need to have some of _Planck_ data to successfully run this code. In particular, you need the [common intensity mask](http://pla.esac.esa.int/pla/\#maps) and [SMICA full-sky map](http://pla.esac.esa.int/pla/\#maps). They should be placed in the `Planck_data` folder, together with the _Planck_'s colormap, which I already provide. Alternatively, you should modify the path to find these data in the `classes.py` file.

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

To perform the analysis it is sufficient to run `complete_pipeline.py`, given that you have all the necessary dependencies and data. Alternatively, if you want to run the analysis on a single $\ell_{\rm max}$ choice, run one of `run_lmax4.py`, `run_lmax6.py` or `run_lmax10.py`.
