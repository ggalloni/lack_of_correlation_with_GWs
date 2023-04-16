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

In order to run the code you also need an implementation of [CLASS](https://github.com/lesgourg/class_public) ([D. Blas et al., 2011](http://arxiv.org/abs/1104.2933)), which encodes the Boltzmann equations for the CGWB, as defined in [A. Ricciardone et al., 2021](https://arxiv.org/pdf/2106.02591.pdf).

## Repeating the analysis

Assuming that you have all the necessary dependencies, data and input spectra, to perform the analysis it is sufficient to run `complete_pipeline.py`. Alternatively, if you want to run the analysis on a single $\ell_{\rm max}$ choice, run one of `run_lmax4.py`, `run_lmax6.py` or `run_lmax10.py`.

## Testing

As an example, I provide the results of a test pipeline, named "testing", assuming $N=100$. In particular, in the folder `data\testing` you can find every product of the analysis, figures included.
