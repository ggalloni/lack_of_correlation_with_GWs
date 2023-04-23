# Scripts

This folder contains the various scripts carrying out the analysis.
 
* cgwb_angular_correlation_function

Compute the angular correlation function $C(\theta)$ for GWGW and TGW. 
In particular, this is the script producing the results in appendix B where we assume
$\ell_{\rm max} = 2000$ to show the angular correlation down to $0.1^\circ$.

* common_functions

Store the functions used in more than one script.

* compute_S_estimators

Take the various $C_{\ell}$ and computes the S_{\theta_{\rm min}, \theta_{\rm max}} estimators.
This is done both for single fields (TT, TGW and GWGW) and for combinations of them.

* compute_legendre_integrals

For each combination of $\ell - \ell'$, compute the integral of the product of the Legendre polynimials
over a specific angular range $[\theta_{\rm min}, \theta_{\rm max}]$.

* compute_legendre_polynomials

Compute the legendre polynomials in a specific angular range. It is only useful 
when computing the angular correlation functions to plot them.

* compute_window_function

Compute the window function of the considered mask and for a specific $\ell_{\rm max}$

* make_tables

After having computed everything, collect the results in the form of tables. 
The inner part of the tables can be copyied and pasted to the Overleaf.

* plot_S_histograms

Plot the histogram of the values of S for the optimal angular range of every field.

* plot_angular_correlation_function

Plot the $\Lambda$CDM 2-points correlation function with the 68% and 95% bands of 
the constrained realizations. This is repeated for every field. In the case of TT, the constrained
realizations are replaced by the values from SMICA.

* plot_cumulative_S_estimators

Plot the histograms of the cumulative S estimator for each field.

* plot_cumulative_significance

Plot the histograms of the significance for each field.

* plot_optimal_angular_ranges

For each field and each angular range, compute the PD. Then it plot each PD in a heatmap for each field.

* save_cumulative_S_estimators

Compute the sum over angular configurations of the S estimators.

* save optimal_angular_ranges

Save the optimal angular range for each field.

* save_significance

Compute the significance of each realization and each field. Then, stores them into files.

* simulations_production

Produce simulations of each field, both full-sky and masked. Here, many things are performed, 
since this script constitute the core of the analysis. 

* spectra_production

Compute the theoretical angular power spectra for each field using a modified version of CLASS.
