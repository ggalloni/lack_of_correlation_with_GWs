<p align="center">
  <img src="https://github.com/ggalloni/lack_of_correlation_with_GWs/blob/main/lack_logo2.png" width="500">
</p>

# Lack-of-Correlation with Gravitational Waves

This repository collects the codes to run the analysis on the lack-of-correlation anomaly using Gravitational Waves (GWs). 
The reference paper is:

Giacomo Galloni, Mario Ballardini, Nicola Bartolo, Alessandro Gruppuso, Luca Pagano, Angelo Ricciardone, ``_Unraveling the CMB lack-of-correlation anomaly with the cosmological gravitational wave background_'', JCAP10(2023)013, arxiv: [2305.18184](https://arxiv.org/abs/2305.18184).

For further details on the analysis refer to that work.

If you use this code, or part of it, please cite the paper above.

## Necessary data

Note that you need to have some of _Planck_ data to successfully run this code. In particular, you need the [common intensity mask](http://pla.esac.esa.int/pla/\#maps) and [SMICA full-sky map](http://pla.esac.esa.int/pla/\#maps). They should be placed in the `Planck_data` folder, together with the _Planck_'s colormap, which I already provided. Alternatively, you should modify the path to find these data in the `GW_lack_corr/src/classes/settings.py` file.

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

Assuming that you have all the necessary dependencies, data, and input spectra, to perform the analysis it is sufficient to run `GW_lack_corr/complete_pipeline.py`. Alternatively, if you want to run the analysis on a single $\ell_{\rm max}$ choice, run one of `GW_lack_corr/individual_runs/run_lmax4.py`, `GW_lack_corr/individual_runs/run_lmax6.py` or `GW_lack_corr/individual_runs/run_lmax10.py`.

## Testing

As an example, I provide the results of a test pipeline, named "testing", assuming $N=100$. In particular, in the folder `data/testing` you can find every product of the analysis, figures included.

## Important notes

To run the actual analysis, I use $N=10000$ simulations. This means that the `GW_lack_corr/complete_pipeline.py` script takes approximately 20 minutes to finish (I have 16 GB of RAM and 8 cores for a total of 16 CPUs). Also, the resulting folder containing the products will occupy approximately 104 GB.
