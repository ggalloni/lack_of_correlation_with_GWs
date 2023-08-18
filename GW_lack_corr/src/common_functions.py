from pathlib import Path
import pickle
import time
from typing import Any

import healpy as hp
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pymaster as nmt
import ray
import scipy.integrate as inte
import scipy.special as ss
import scipy.stats as stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm
from scipy.special import sph_harm as Ylm
from scipy.integrate import nquad

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter("ignore", category=AstropyWarning)
np.seterr(divide="ignore", invalid="ignore")


def integrate_spherical_harmonics(l1: int, m1: int, l2: int, m2: int) -> float:
    """
    Compute the integral of the product of two spherical harmonics with
    rank (l1, m1) and (l2, m2) over the sphere, and compare it to the expected
    value of 1 if l1=l2 and m1=m2, and 0 otherwise.

    Parameters
    ----------
    l1 : int
        Rank of the first spherical harmonic.
    m1 : int
        Orientation index of the first spherical harmonic.
    l2 : int
        Rank of the second spherical harmonic.
    m2 : int
        Orientation index of the second spherical harmonic.

    Returns
    -------
    integral : float
        Integral of the product of the two spherical harmonics over the
        sphere.
    expected : float
        Expected value of the integral (1 if l1=l2 and m1=m2, 0 otherwise)
    """

    def integrand(theta, phi):
        Y1 = Ylm(m1, l1, phi, theta)
        Y2 = Ylm(m2, l2, phi, theta)
        return (
            np.sin(theta) * (Y1 * np.conj(Y2)).real
        )  # Only real part here (easier to integrate)

    integral, _ = nquad(integrand, [[0, np.pi], [0, 2 * np.pi]])

    expected = float((l1 == l2) and (m1 == m2))

    return integral, expected


def integrate_spherical_harmonics_pix(
    l1: int, m1: int, l2: int, m2: int, nside: int
) -> float:
    """
    Compute the integral of the product of two spherical harmonics with
    rank (l1, m1) and (l2, m2) over the pixels of a HEALPix partition with
    resolution nside, and compare it to the expected value of 1 if l1=l2 and
    m1=m2, and 0 otherwise.

    Parameters
    ----------
    l1 : int
        Rank of the first spherical harmonic.
    m1 : int
        Orientation index of the first spherical harmonic.
    l2 : int
        Rank of the second spherical harmonic.
    m2 : int
        Orientation index of the second spherical harmonic.
    nside : int
        Resolution of the HEALPix partition.

    Returns
    -------
    integral : float
        Integral of the product of the two spherical harmonics over the
        HEALPix partition.
    expected : float
        Expected value of the integral (1 if l1=l2 and m1=m2, 0 otherwise)
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    def integrand(ipix):
        Y1 = Ylm(m1, l1, phi[ipix], theta[ipix])
        Y2 = Ylm(m2, l2, phi[ipix], theta[ipix])
        return Y1 * np.conj(Y2)

    integral = np.sum(integrand(np.arange(npix)))
    integral *= hp.nside2pixarea(nside)

    expected = float((l1 == l2) and (m1 == m2))

    return integral, expected


def integrate_spherical_harmonics_pix_mask(
    l1: int, m1: int, l2: int, m2: int, nside: int, mask: np.ndarray
) -> float:
    """
    Compute the integral of the product of two spherical harmonics with
    rank (l1, m1) and (l2, m2) over the pixels of a HEALPix partition with
    resolution nside that are marked as valid by the mask, and
    compare it to the expected value of 1 if l1=l2 and m1=m2, and 0 otherwise.

    Parameters
    ----------
    l1 : int
        Rank of the first spherical harmonic.
    m1 : int
        Orientation index of the first spherical harmonic.
    l2 : int
        Rank of the second spherical harmonic.
    m2 : int
        Orientation index of the second spherical harmonic.
    nside : int
        Resolution of the HEALPix partition.
    mask : np.ndarray
        Mask of the CMB sky, with 1 for valid pixels and 0 for invalid ones.

    Returns
    -------
    integral : float
        Integral of the product of the two spherical harmonics over the
        HEALPix partition.
    expected : float
        Expected value of the integral (1 if l1=l2 and m1=m2, 0 otherwise)
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    valid_indices = np.where(mask == 1)[0]

    def integrand(ipix):
        Y1 = Ylm(m1, l1, phi[ipix], theta[ipix])
        Y2 = Ylm(m2, l2, phi[ipix], theta[ipix])
        return Y1 * np.conj(Y2)

    integral = np.sum(integrand(valid_indices))
    integral *= hp.nside2pixarea(nside)

    expected = float((l1 == l2) and (m1 == m2))

    return integral, expected


def build_window_function(lmax: int, nside: int, mask: np.ndarray) -> np.ndarray:
    """Build the window function matrix for a given mask.

    Parameters
    ----------
    lmax : int
        The maximum multipole of the window function.
    nside : int
        The resolution of the HEALPix partition.
    mask : np.ndarray
        The mask of the CMB sky, with 1 for valid pixels and 0 for invalid ones.

    Returns
    -------
    W : np.ndarray
        The window function matrix.
    """
    alm_size = hp.Alm.getsize(lmax)

    W = np.zeros((alm_size, alm_size), dtype=complex)
    for i in range(alm_size):
        l1, m1 = hp.Alm.getlm(lmax, i)
        for j in range(i, alm_size):
            l2, m2 = hp.Alm.getlm(lmax, j)
            W[i, j], _ = integrate_spherical_harmonics_pix_mask(
                l1, m1, l2, m2, nside, mask
            )
            W[j, i] = W[i, j]

    return W


# def correct_mat(mat, lmax):
#     """Correct a matrix for cross-correlation.

#     Parameters
#     ----------
#     mat : np.ndarray
#         The matrix to correct.
#     lmax : int
#         The maximum multipole of the matrix.
#     """
#     if mat.shape[0] != mat.shape[1]:
#         raise ValueError("The matrix is not square.")
#     mat[1:, 1:] *= np.sqrt(1)
#     mat[1:, 1:] *= np.sqrt(4)
#     mat[0, :] = 0
#     mat[:, 0] = 0
#     mat[1, :] = 0
#     mat[:, 1] = 0
#     mat[lmax + 1, :] = 0
#     mat[:, lmax + 1] = 0
#     return mat


def decouple_alms(
    alms: np.ndarray, real_cov: np.ndarray, imag_cov: np.ndarray, CLS: dict, lmax: int
) -> np.ndarray:
    """Decouple the alms from the GW and noise covariance.

    Parameters
    ----------
    alms : np.ndarray
        The alms of the map we want to decouple.
    real_cov : np.ndarray
        The real part of the GW and noise covariance matrix.
    imag_cov : np.ndarray
        The imaginary part of the GW and noise covariance matrix.
    CLS : dict
        The power spectra of the map.
    lmax : int
        The maximum l value to use in the decoupling.

    Returns
    -------
    decoupled_alm : np.ndarray
        The decoupled alms.
    """
    # Decouple the real part of the map
    decoupled_alm_real = np.zeros(alms.shape, dtype=complex)
    decoupled_alm_real[2 : lmax + 1] = (
        np.linalg.inv(np.linalg.cholesky(real_cov[2 : lmax + 1, 2 : lmax + 1]))
        @ alms[2 : lmax + 1].real
    )
    decoupled_alm_real[lmax + 2 :] = (
        np.linalg.inv(np.linalg.cholesky(real_cov[lmax + 2 :, lmax + 2 :]))
        @ alms[lmax + 2 :].real
    )

    # Decouple the imaginary part of the map
    decoupled_alm_imag = np.zeros(alms.shape, dtype=complex)
    decoupled_alm_imag[lmax + 2 :] = (
        np.linalg.inv(np.linalg.cholesky(imag_cov[lmax + 2 :, lmax + 2 :]))
        @ alms[lmax + 2 :].imag
    )

    # Combine the two parts of the decoupled alms
    decoupled_alm = decoupled_alm_real + decoupled_alm_imag * 1j

    # Normalize the alms
    decoupled_alm = hp.almxfl(
        decoupled_alm, CLS["tgw"][: lmax + 1] / np.sqrt(CLS["tt"][: lmax + 1])
    )
    decoupled_alm[np.isnan(decoupled_alm)] = 0
    decoupled_alm[np.isinf(decoupled_alm)] = 0
    return decoupled_alm


@ray.remote
def mask_TT_un(N: int, cmb_real: dict, mask: np.ndarray, lmax: int, nside: int) -> list:
    """Calculate the TT power spectrum of the realisations of the CMB using a given mask.

    Parameters
    ----------
    N : int
        The number of realisations of the CMB.
    cmb_real : dict
        The realisations of the CMB.
    mask : np.ndarray
        The mask used to calculate the TT power spectrum.
    lmax : int
        The maximum multipole of the TT power spectrum.
    nside : int
        The resolution of the TT power spectrum.

    Returns
    -------
    res : list
        The TT power spectrum of the realisations of the CMB.
    """
    res = []
    for i in range(N):
        f_T = nmt.NmtField(mask, [hp.alm2map(cmb_real["%s" % i], nside)], spin=0)
        if i == 0:
            b = nmt.NmtBin.from_nside_linear(nside, 1)
            wsp = nmt.NmtWorkspace()
            wsp.compute_coupling_matrix(f_T, f_T, b, is_teb=False)
        res.append(
            np.insert(
                nmt.compute_full_master(f_T, f_T, b, workspace=wsp)[0], [0, 0], [0, 0]
            )[: lmax + 1]
        )
    return res


@ray.remote
def mask_X_un(
    N: int,
    cmb_real: dict,
    mask: np.ndarray,
    lmax: int,
    nside: int,
    inv_window: np.ndarray,
    CLs: dict,
    seeds: list,
) -> list:
    """Calculate the TGW power spectrum of the realisations of the CGWB using a given mask.

    Parameters
    ----------
    N : int
        The number of realisations of the CMB.
    cmb_real : dict
        The realisations of the CMB.
    mask : np.ndarray
        The mask used to calculate the GW power spectrum.
    lmax : int
        The maximum multipole of the GW power spectrum.
    nside : int
        The resolution of the GW power spectrum.
    inv_window : np.ndarray
        The inverse of the window function.
    CLs : dict
        The power spectra of the map.
    seeds : list
        The seeds used to generate the realisations of the CGWB.

    Returns
    -------
    res : list
        The TGW power spectrum of the realisations of the CGWB.
    """
    res = []
    filt = np.ones(lmax + 1)
    for i in range(N):
        f_T = nmt.NmtField(mask, [hp.alm2map(cmb_real["%s" % i], nside)], spin=0)
        masked_alm = f_T.get_alms()[0]
        masked_alm = hp.almxfl(masked_alm, filt)
        masked_alm = masked_alm[masked_alm != 0]
        decoupled_alm = masked_alm @ inv_window
        seed = hp.almxfl(decoupled_alm, CLs["tgw"][: lmax + 1] / CLs["tt"][: lmax + 1])
        seed[np.isnan(seed)] = 0
        np.random.seed(seeds[i])  # noqa: NPY002
        cgwb_real = hp.synalm(CLs["constrained_gwgw"], lmax=lmax) + seed
        f_GW = nmt.NmtField(np.ones(mask.shape), [hp.alm2map(cgwb_real, nside)], spin=0)
        if i == 0:
            b = nmt.NmtBin.from_nside_linear(nside, 1)
            wsp = nmt.NmtWorkspace()
            wsp.compute_coupling_matrix(f_T, f_GW, b, is_teb=False)
        res.append(
            np.insert(
                nmt.compute_full_master(f_T, f_GW, b, workspace=wsp)[0], [0, 0], [0, 0]
            )[: lmax + 1]
        )
    return res


@ray.remote
def mask_X_con(
    N: int, cmb_map: dict, cgwb_real: dict, mask: np.ndarray, lmax: int, nside: int
) -> list:
    """Calculate the TGW power spectrum of the realisations of the GWB using a given mask and a CMB skymap.

    Parameters
    ----------
    N : int
        The number of realisations of the CMB.
    cmb_real : dict
        The realisations of the CMB.
    mask : np.ndarray
        The mask used to calculate the GW power spectrum.
    lmax : int
        The maximum multipole of the GW power spectrum.
    nside : int
        The resolution of the GW power spectrum.

    Returns
    -------
    res : list
        The TGW power spectrum of the realisations of the CGWB.
    """
    res = []
    for i in range(N):
        f_T = nmt.NmtField(mask, [cmb_map], spin=0)
        f_GW = nmt.NmtField(
            np.ones(mask.shape), [hp.alm2map(cgwb_real["%s" % i], nside)], spin=0
        )
        if i == 0:
            b = nmt.NmtBin.from_nside_linear(nside, 1)
            wsp = nmt.NmtWorkspace()
            wsp.compute_coupling_matrix(f_T, f_GW, b, is_teb=False)
        res.append(
            np.insert(
                nmt.compute_full_master(f_T, f_GW, b, workspace=wsp)[0], [0, 0], [0, 0]
            )[: lmax + 1]
        )
    return res


@ray.remote
def get_sample_field(
    i: int, real: dict, mask: np.ndarray, filter: np.ndarray, nside: int
) -> np.ndarray:
    """Get the sample field for the i-th realization of the CGWB.

    Parameters
    ----------
    i : int
        The i-th realization of the CGWB.
    real : dict
        The realizations of the CGWB.
    mask : np.ndarray
        The mask used to calculate the GW power spectrum.
    filter : np.ndarray
        The filter used to calculate the GW power spectrum.
    nside : int
        The resolution of the GW power spectrum.

    Returns
    -------
    res : np.ndarray
        The sample field for the i-th realization of the CGWB.
    """

    # Get the field
    f_T = nmt.NmtField(mask, [hp.alm2map(real[i], nside)])
    # Get the alms
    res = f_T.get_alms()[0]
    # Apply the filter
    res = hp.almxfl(res, filter)
    # Return only the non-zero elements
    return res[res != 0]


@ray.remote
def cov_comp_real(k: int, realizations: np.ndarray) -> np.ndarray:
    """Compute the covariances for the complex numbers in the k-th row of realizations.

    Parameters
    ----------
    k: int
        The row of realizations to compute covariances for.
    realizations: np.ndarray
        The array of complex numbers to compute covariances for.

    Returns
    -------
    Covariance: np.ndarray
        The computed covariances.
    """
    covariances = np.zeros((len(realizations[0]), len(realizations[0])))
    for i in range(len(realizations[0])):
        for j in range(len(realizations[0])):
            covariances[i, j] = (realizations[k][i] * realizations[k][j]).real
    return np.real(covariances)


@ray.remote
def cov_comp_imag(k: int, realizations: np.ndarray) -> np.ndarray:
    """Compute the covariances for the complex numbers in the k-th row of realizations.

    Parameters
    ----------
    k: int
        The row of realizations to compute covariances for.
    realizations: np.ndarray
        The array of complex numbers to compute covariances for.

    Returns
    -------
    Covariance: np.ndarray
        The computed covariances.
    """
    covariances = np.zeros((len(realizations[0]), len(realizations[0])))
    for i in range(len(realizations[0])):
        for j in range(len(realizations[0])):
            covariances[i, j] = (realizations[k][i] * realizations[k][j].conj()).imag
    return np.real(covariances)


def legendre_intergal(μ_min: np.ndarray, μ_max: np.ndarray, lmax: int) -> np.ndarray:
    """Calculate the integral of the Legendre polynomials.

    Parameters
    ----------
    μ_min : np.ndarray
        The minimum value of the cosine of the angle between the vector and the line of sight.
    μ_max : np.ndarray
        The maximum value of the cosine of the angle between the vector and the line of sight.
    lmax : int
        The maximum multipole of the GW power spectrum.

    Returns
    -------
    res : np.ndarray
        The integral of the Legendre polynomials.
    """

    # initialize the result matrix
    res = np.zeros(shape=(len(μ_min), len(μ_max), lmax + 1 - 2, lmax + 1 - 2))

    # loop over the different values of μ_min
    for angmin in nb.prange(len(μ_min)):
        # compute the cosine of the current μ_min
        cos_min = np.cos(np.deg2rad(μ_min[angmin]))

        # loop over the different values of μ_max
        for angmax in nb.prange(len(μ_max)):
            # compute the cosine of the current μ_max
            cos_max = np.cos(np.deg2rad(μ_max[angmax]))

            # check if the current μ_max is greater than the current μ_min
            if angmin < angmax:
                # loop over the different values of l
                for ell in nb.prange(2, lmax + 1, 1):
                    # loop over the different values of k
                    for ellp in nb.prange(2, lmax + 1, 1):
                        # compute the integral
                        res[angmin, angmax, ell - 2, ellp - 2] = inte.quad(
                            lambda x: ss.eval_legendre(ell, x)
                            * ss.eval_legendre(ellp, x),
                            cos_max,
                            cos_min,
                        )[0]
                        # copy the result to the other half of the matrix
                        res[angmax, angmin, ell - 2, ellp - 2] = res[
                            angmin, angmax, ell - 2, ellp - 2
                        ]

    return res


@nb.njit(parallel=True)
def S_estimator(
    μ_min: np.ndarray,
    μ_max: np.ndarray,
    integral: np.ndarray,
    CLS: np.ndarray,
    N: int,
    lmax: int,
) -> np.ndarray:
    """Estimate the S estimators.

    Parameters
    ----------
    μ_min : np.ndarray
        The minimum value of the cosine of the angle between the vector and the line of sight.
    μ_max : np.ndarray
        The maximum value of the cosine of the angle between the vector and the line of sight.
    integral : np.ndarray
        The result of the integral of the Legendre polynomials.
    CLS : np.ndarray
        The power spectrum of the CGWB.
    N : int
        The number of realizations.
    lmax : int
        The maximum multipole of the GW power spectrum.

    Returns
    -------
    S_x : np.ndarray
        The S estimator value.
    """
    # Define a matrix to store the result
    S_x = np.zeros(shape=(len(μ_min), len(μ_max), N))
    # Define the ell values
    ells = np.arange(0, lmax + 1, 1)
    # Loop over realizations
    for i in nb.prange(N):
        # Load the CLS for this realization
        CLS_i = CLS[i]
        # Define a matrix to store the intermediate result
        temp = np.zeros(shape=(lmax + 1 - 2, lmax + 1 - 2))
        # Loop over μ_min
        for k in nb.prange(len(μ_min)):
            # Loop over μ_max
            for m in nb.prange(len(μ_max)):
                # Only compute the result if μ_min < μ_max
                if k < m:
                    # Load the intermediate result for this realization
                    I_km = integral[k, m, :, :]
                    # Loop over l
                    for h in nb.prange(2, lmax + 1, 1):
                        # Loop over j
                        for j in nb.prange(2, lmax + 1, 1):
                            # Compute the intermediate result
                            temp[h - 2, j - 2] = (
                                (2 * ells[h] + 1)
                                / 4
                                / np.pi
                                * (2 * ells[j] + 1)
                                / 4
                                / np.pi
                                * CLS_i[h]
                                * I_km[h - 2, j - 2]
                                * CLS_i[j]
                            )
                    # Compute the result
                    S_x[k, m, i] = np.sum(temp)
                    # Store the result symmetrically
                    S_x[m, k, i] = S_x[k, m, i]
    return S_x


# @ray.remote
def compute_S(
    file: str,
    μ_min: np.ndarray,
    μ_max: np.ndarray,
    integral: np.ndarray,
    TT_uncon_CL: np.ndarray,
    N: int,
    lmax: int,
) -> None:
    """Compute the S estimators.

    Parameters
    ----------
    file : str
        The output file.
    μ_min : np.ndarray
        The minimum value of the cosine of the angle between the vector and the line of sight.
    μ_max : np.ndarray
        The maximum value of the cosine of the angle between the vector and the line of sight.
    integral : np.ndarray
        The result of the integral of the Legendre polynomials.
    TT_uncon_CL : np.ndarray
        The power spectrum of the CGWB.
    N : int
        The number of realizations.
    lmax : int
        The maximum multipole of the GW power spectrum.
    """
    start = time.time()
    S = np.array(S_estimator(μ_min, μ_max, integral, TT_uncon_CL, N, lmax))
    with open(file, "wb") as pickle_file:
        pickle.dump(S, pickle_file)
    S = 0
    end = time.time()
    print(f"Done in {round(end-start, 2)} seconds! Puff...")
    return


# @ray.remote
def compute_joint_S(
    out: str, T_file: str, con_T_file: str, GW_file: str, con_GW_file: str
) -> None:
    """Compute the joint S estimators combining single field ones.

    Parameters
    ----------
    out : str
        The output file.
    T_file : str
        The file containing the TT S estimators.
    con_T_file : str
        The file containing the TT S estimators for the constrained case.
    GW_file : str
        The file containing the GWGW S estimators.
    con_GW_file : str
        The file containing the GWGW S estimators for the constrained case.
    """

    start = time.time()

    with open(T_file, "rb") as pickle_file:
        uncon_TT = pickle.load(pickle_file)
    with open(GW_file, "rb") as pickle_file:
        uncon_GWGW = pickle.load(pickle_file)
    uncon_TT = np.mean(uncon_TT, axis=2)[:, :, np.newaxis]
    uncon_GWGW = np.mean(uncon_GWGW, axis=2)[:, :, np.newaxis]

    with open(con_T_file, "rb") as pickle_file:
        con_TT = pickle.load(pickle_file)
    with open(con_GW_file, "rb") as pickle_file:
        con_GWGW = pickle.load(pickle_file)
    if isinstance(con_TT[0, 0], np.float64):
        con_TT = con_TT[:, :, np.newaxis]
    con_TT = (con_TT / uncon_TT) ** 2
    con_TT[np.isnan(con_TT)] = 0
    uncon_TT = 0
    con_GWGW = (con_GWGW / uncon_GWGW) ** 2
    con_GWGW[np.isnan(con_GWGW)] = 0
    uncon_GWGW = 0

    with open(out, "wb") as pickle_file:
        pickle.dump(np.sqrt(con_TT + con_GWGW), pickle_file)

    end = time.time()
    print(f"Done in {round(end-start, 2)} seconds! Puff...")
    return


# @ray.remote
def compute_super_S(
    out: str,
    T_file: str,
    con_T_file: str,
    GW_file: str,
    con_GW_file: str,
    X_file: str,
    con_X_file: str,
) -> None:
    """Compute the joint S estimators combining single field ones.

    Parameters
    ----------
    out : str
        The output file.
    T_file : str
        The file containing the TT S estimators.
    con_T_file : str
        The file containing the TT S estimators for the constrained case.
    GW_file : str
        The file containing the GWGW S estimators.
    con_GW_file : str
        The file containing the GWGW S estimators for the constrained case.
    X_file : str
        The file containing the TGW S estimators.
    con_X_file : str
        The file containing the TGW S estimators for the constrained case.
    """

    start = time.time()

    with open(T_file, "rb") as pickle_file:
        uncon_TT = pickle.load(pickle_file)
    uncon_TT = np.mean(uncon_TT, axis=2)[:, :, np.newaxis]
    with open(con_T_file, "rb") as pickle_file:
        con_TT = pickle.load(pickle_file)
    if isinstance(con_TT[0, 0], np.float64):
        con_TT = con_TT[:, :, np.newaxis]
    con_TT = (con_TT / uncon_TT) ** 2
    con_TT[np.isnan(con_TT)] = 0
    uncon_TT = 0

    with open(GW_file, "rb") as pickle_file:
        uncon_GWGW = pickle.load(pickle_file)
    uncon_GWGW = np.mean(uncon_GWGW, axis=2)[:, :, np.newaxis]
    with open(con_GW_file, "rb") as pickle_file:
        con_GWGW = pickle.load(pickle_file)
    con_GWGW = (con_GWGW / uncon_GWGW) ** 2
    con_GWGW[np.isnan(con_GWGW)] = 0
    uncon_GWGW = 0

    with open(X_file, "rb") as pickle_file:
        uncon_X = pickle.load(pickle_file)
    uncon_X = np.mean(uncon_X, axis=2)[:, :, np.newaxis]
    with open(con_X_file, "rb") as pickle_file:
        con_X = pickle.load(pickle_file)
    con_X = (con_X / uncon_X) ** 2
    con_X[np.isnan(con_X)] = 0
    uncon_X = 0

    with open(out, "wb") as pickle_file:
        pickle.dump(np.sqrt(con_TT + con_X + con_GWGW), pickle_file)

    end = time.time()
    print(f"Done in {round(end-start, 2)} seconds! Puff...")
    return


def save_optimal(
    uncon_file: str,
    con_file: str,
    lmax: int,
    step: int = 1,
    thrshld: float = 99.5,
    rounding: int = int(0.01 * 100),
) -> list:
    """Find and save the optimal angular ranges.

    Parameters
    ----------
    uncon_file : str
        The file containing the unconstrained S estimators.
    con_file : str
        The file containing the constrained S estimators.
    lmax : int
        The maximum multipole of the S estimators.
    step : int, optional
        Incremental step of the angles, by default 1
    thrshld : float, optional
        The threshold of the percentiles, by default 99.5
    rounding : int, optional
        The rounding of the PDs obtained, by default int(0.01 * 100)

    Returns
    -------
    list
        The list containing the optimal angular ranges.
    """

    with open(uncon_file, "rb") as pickle_file:
        uncon = pickle.load(pickle_file)
    with open(con_file, "rb") as pickle_file:
        con = pickle.load(pickle_file)

    if len(con[0, 0]) == 1:
        con = con[:, :, np.newaxis]

    start = int(180 / (lmax - 1))
    μ_min = np.arange(start, 180 + step, step)
    μ_max = np.arange(start, 180 + step, step)

    bounds = np.zeros((len(μ_min), len(μ_max)))
    for i in range(len(μ_min)):
        for j in range(len(μ_max)):
            if i < j:
                bounds[i, j] = np.percentile(con[i, j, :], thrshld)
                bounds[j, i] = bounds[i, j]
    con = 0

    counts = np.zeros((len(μ_min), len(μ_max)))
    for i in range(len(μ_min)):
        for j in range(len(μ_max)):
            if i < j:
                counts[i, j] = np.mean(uncon[i, j, :] > bounds[i, j]) * 100
                counts[j, i] = counts[i, j]
    uncon = 0

    counts = rounding * np.ndarray.round(counts * 100 / rounding) / 100

    opt_region = np.where(counts == np.max(counts))
    idx = np.argmax(opt_region[1] - opt_region[0])
    idx_min, idx_max = opt_region[0][idx], opt_region[1][idx]

    return [μ_min[idx_min], μ_max[idx_max], np.max(counts)]


def optimal_ang(  # noqa: C901
    uncon_file: str,
    con_file: str,
    lmax: int,
    field: str,
    lab: str,
    cmap: str,
    fig_dir: str,
    *,
    step: int = 1,
    thrshld: float = 99.5,
    rounding: int = int(0.01 * 100),
    step_mesh: float = 0.05,
    savefig: bool = False,
    show: bool = True,
) -> None:
    """Plot the optimal angular ranges.

    Parameters
    ----------
    uncon_file : str
        The file containing the unconstrained S estimators.
    con_file : str
        The file containing the constrained S estimators.
    lmax : int
        The maximum multipole of the S estimators.
    field : str
        The field to plot.
    lab : str
        The label of the field.
    cmap : str
        The colormap to use.
    fig_dir : str
        The directory to save the figure.
    step : int, optional
        Incremental step of the angles, by default 1
    thrshld : float, optional
        The threshold of the percentiles, by default 99.5
    rounding : int, optional
        The rounding of the PDs obtained, by default int(0.01 * 100)
    step_mesh : float, optional
        The step of the meshgrid, by default 0.05
    savefig : bool, optional
        Whether to save the figure, by default False
    show : bool, optional
        Whether to show the figure, by default True
    """
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    with open(uncon_file, "rb") as pickle_file:
        uncon = pickle.load(pickle_file)
    with open(con_file, "rb") as pickle_file:
        con = pickle.load(pickle_file)

    if len(con[0, 0]) == 1:
        con = con[:, :, np.newaxis]

    start = int(180 / (lmax - 1))
    μ_min = np.arange(start, 180 + step, step)
    μ_max = np.arange(start, 180 + step, step)

    bounds = np.zeros((len(μ_min), len(μ_max)))
    for i in range(len(μ_min)):
        for j in range(len(μ_max)):
            if i < j:
                bounds[i, j] = np.percentile(con[i, j, :], thrshld)
                bounds[j, i] = bounds[i, j]
    con = 0

    counts = np.zeros((len(μ_min), len(μ_max)))
    for i in range(len(μ_min)):
        for j in range(len(μ_max)):
            if i < j:
                counts[i, j] = np.mean(uncon[i, j, :] > bounds[i, j]) * 100
                counts[j, i] = counts[i, j]
    uncon = 0

    counts = rounding * np.ndarray.round(counts * 100 / rounding) / 100

    opt_region = np.where(counts == np.max(counts))
    idx = np.argmax(opt_region[1] - opt_region[0])
    idx_min, idx_max = opt_region[0][idx], opt_region[1][idx]

    print("The maximal difference is = ", np.max(counts))
    print("It is at index_min = ", idx_min, "and index_max = ", idx_max)
    print("They correspond to μ_min = ", μ_min[idx_min], "and μ_max = ", μ_max[idx_max])
    print("They also correspond to a value of = ", np.log10(bounds[idx_min, idx_max]))

    plt.scatter(
        μ_max[idx_max],
        μ_min[idx_min],
        marker="*",
        facecolors="black",
        color="white",
        s=150,
        zorder=20,
        label=r"$\theta_{\rm min} = %s^\circ, \theta_{\rm max} = %s^\circ$"
        % (μ_min[idx_min], μ_max[idx_max]),
    )

    unphysical = np.zeros((len(μ_min), len(μ_max)))
    # Mask the pixels above the anti-diagonal
    for i in range(len(μ_min)):
        for j in range(i + 1, len(μ_max)):
            unphysical[i, j] = np.nan
    shaded_data = np.ma.masked_where(np.isnan(unphysical), unphysical)
    cmap_shaded = plt.get_cmap("gray")  # Choose your desired colormap for shading
    plt.pcolormesh(
        μ_min,
        μ_max,
        shaded_data,
        cmap=cmap_shaded,
        alpha=0.2,
        zorder=500,
    )  # Adjust alpha for shading intensity

    plt.pcolormesh(
        μ_min,
        μ_max,
        counts,
        shading="nearest",
        cmap=cmap,
        norm=mpl.colors.BoundaryNorm(
            np.arange(
                np.floor(np.max(counts)) - 5,
                min(100 + step_mesh / 2, np.round(np.max(counts), 1) + step_mesh),
                step_mesh,
            ),
            ncolors=cmap.N,
            clip=False,
        ),
    )
    bounds = 0
    counts = 0

    cbar = plt.colorbar(extend="min")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(
        rf"PD with $S^{{\rm {lab}}}_{{\theta_{{\rm min}}, \theta_{{\rm max}}}}$ [%]",
        fontsize=15,
    )
    plt.ylabel(r"$\theta_{\rm min}$", fontsize=17)
    plt.xlabel(r"$\theta_{\rm max}$", fontsize=17)

    ax.tick_params(
        direction="in", which="both", labelsize=14, width=1.0, length=4, zorder=101
    )
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")

    plt.xlim([None, 180])
    plt.ylim([None, 180])
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath("opt_ang_" + field + f"_lmax{lmax}.png"), dpi=150)
    if show:
        plt.show()
    return


def S_val(
    uncon_file: str,
    lmax: int,
    field: str,
    lab: str,
    cmap: LinearSegmentedColormap,
    fig_dir: Path,
    *,
    unit: str = "",
    vmin: bool = False,
    step: int = 1,
    savefig: bool = False,
    show: bool = True,
) -> None:
    """Plot the values of S for a given field.

    Parameters
    ----------
    uncon_file : str
        The path to the pickle file of the unconstrained S values.
    lmax : int
        The maximum l value.
    field : str
        The field to consider.
    lab : str
        The label of the field.
    cmap : matplotlib.colors.LinearSegmentedColormap
        The colormap.
    fig_dir : pathlib.Path
        The directory to save the figure.
    unit : str, optional
        The unit of the field, by default ""
    vmin : bool, optional
        Whether to set the minimum value of the colorbar to 85% of the maximum value, by default False
    step : int, optional
        The step of the meshgrid, by default 1
    savefig : bool, optional
        Whether to save the figure, by default False
    show : bool, optional
        Whether to show the figure, by default True
    """
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    with open(uncon_file, "rb") as pickle_file:
        uncon = pickle.load(pickle_file)

    start = int(180 / (lmax - 1))
    μ_min = np.arange(start, 180 + step, step)
    μ_max = np.arange(start, 180 + step, step)

    counts = np.zeros((len(μ_min), len(μ_max)))
    for i in range(len(μ_min)):
        for j in range(len(μ_max)):
            if i < j:
                counts[i, j] = np.mean(uncon[i, j, :])
                counts[j, i] = counts[i, j]
    uncon = 0

    if vmin:
        vmin = np.max(counts) * 0.85
    else:
        vmin = np.min(counts)
    plt.pcolormesh(
        μ_min,
        μ_max,
        counts,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
    )
    counts = 0

    cbar = plt.colorbar(extend="min")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(
        rf"$\langle S^{{\rm {lab}}}_{{\theta_{{\rm min}},\theta_{{\rm max}}}}\rangle$ "
        + unit,
        fontsize=15,
    )
    plt.ylabel(r"$\theta_{\rm min}$", fontsize=17)
    plt.xlabel(r"$\theta_{\rm max}$", fontsize=17)

    ax.tick_params(
        direction="in", which="both", labelsize=14, width=1.0, length=4, zorder=101
    )
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")

    plt.xlim([None, 180])
    plt.ylim([None, 180])
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath("S_val_" + field + f"_lmax{lmax}.png"), dpi=150)
    if show:
        plt.show()
    return


def summing(
    out: str, file_uncon: str, file_con: str, mask: np.ndarray, *, norm: bool = True
) -> None:
    """Sum the S estimators over angular configurations.

    Parameters
    ----------
    out : str
        The path to the output directory.
    file_uncon : str
        The path to the unconstrained pickle file.
    file_con : str
        The path to the constrained pickle file.
    mask : np.ndarray
        The mask of the field.
    norm : bool, optional
        Whether to normalize the constrained array by the mean of the unconstrained array, by default True
    """

    start = time.time()
    if file_uncon == file_con:
        with open(file_uncon, "rb") as pickle_file:
            uncon_arr = pickle.load(pickle_file)
        con_arr = uncon_arr[:]
    else:
        with open(file_uncon, "rb") as pickle_file:
            uncon_arr = pickle.load(pickle_file)
        with open(file_con, "rb") as pickle_file:
            con_arr = pickle.load(pickle_file)
    if norm:
        mean_arr = np.mean(uncon_arr, axis=2)
        con_arr = con_arr / mean_arr[:, :, np.newaxis]
        if len(con_arr[0, 0]) == 1:
            con_arr = np.ma.masked_array(con_arr, mask=mask[:, :, 0])
        else:
            con_arr = np.ma.masked_array(con_arr, mask=mask)
    con_arr[np.isnan(con_arr)] = 0
    con_arr = np.sum(con_arr, axis=(0, 1))

    with open(out, "wb") as pickle_file:
        pickle.dump(con_arr, pickle_file)
    end = time.time()
    print(f"Summing took {round(end - start, 2)} seconds.")
    return


def plot_gauss(
    data: np.ndarray,
    cmb: float,
    *,
    label: str = "Insert Label",
    verbose: bool = False,
    **kwargs: dict,
) -> None:
    """Plot Gaussian fit of some data.

    Parameters
    ----------
    data : np.ndarray
        The data to plot.
    cmb : float
        The value of the CMB.
    label : str, optional
        The label of the plot, by default "Insert Label"
    verbose : bool, optional
        Whether to print the mean, sigma and compatibility, by default False
    **kwargs : dict
        The matplotlib kwargs.
    """

    (mu, sigma) = norm.fit(data)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)

    plt.plot(x, y, label=label, **kwargs)

    if verbose:
        print(
            "Sigma CV = ",
            sigma,
            "; Mean = ",
            mu,
            "; Compatibility = ",
            np.abs(mu - cmb) / sigma,
            ".",
        )

    return


def count(array: np.ndarray, obs: str = "OBSERVABLE", ref: float = None) -> None:
    """Count the number of elements above 3 sigma and above a certain reference value.

    Parameters
    ----------
    array : np.ndarray
        The array to count.
    obs : str, optional
        The name of the observable, by default "OBSERVABLE"
    ref : float, optional
        The reference value, by default None
    """
    if ref is None:
        print(
            rf"*--> For {obs}, the {round(np.mean(array > 3) * 100, 2)}% is above 3 sigma"
        )
    else:
        if ref > 2:
            print(
                rf"*--> For {obs}, the {round(np.mean(array > ref) * 100, 2)}% is above Masked SMICA ({round(ref, 3)} sigma)"
            )
        else:
            print(
                rf"*--> For {obs}, the {round(np.mean(array > ref) * 100, 2)}% is above SMICA ({round(ref, 3)} sigma)"
            )

    lower = round(np.percentile(array, 2.5), 2)
    upper = round(np.percentile(array, 97.5), 2)
    print(rf"*--> For {obs}, the mean is {round(np.mean(array), 2)}")
    print(rf"*--> For {obs}, the 95% CL is [{lower}, {upper}]")

    return


def read(file: str) -> Any:
    """Read a pickle file.

    Returns
    -------
    Any
        The content of the pickle file.
    """
    with open(file, "rb") as pickle_file:
        return pickle.load(pickle_file)


def plot_cumulative_S(
    uncon: np.ndarray,
    con: np.ndarray,
    masked_uncon: np.ndarray,
    masked_con: np.ndarray,
    nbins: int,
    lmax: int,
    field: str,
    lab: str,
    fig_dir: str,
    *,
    unit: str = "",
    concl: bool = False,
    savefig: bool = False,
) -> None:
    """Plot the cumulative S estimators for a certain field.

    Parameters
    ----------
    uncon : np.ndarray
        Unconstrained realizations.
    con : np.ndarray
        Constrained realizations.
    masked_uncon : np.ndarray
        Unconstrained realizations with masked pixels.
    masked_con : np.ndarray
        Constrained realizations with masked pixels.
    nbins : int
        Number of bins.
    lmax : int
        Maximum multipole.
    field : str
        Field name.
    lab : str
        Label for the field.
    fig_dir : str
        Directory where to save figures.
    unit : str, optional
        Unit of the field.
    concl : bool, optional
        If True, add conclusion to the figure in the form of shaded area corresponding to different conclusions.
    savefig : bool, optional
        If True, save figure.
    """

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.set(xscale="log", yscale="linear")
    ax2 = ax.twiny()

    ax2.hist(
        np.log10(uncon),
        bins=nbins,
        histtype="step",
        lw=2,
        color="red",
        density=True,
        label="Unconstr.",
    )
    lower_uncon = np.percentile(np.log10(uncon), 0.5)
    upper_uncon = np.percentile(np.log10(uncon), 99.5)

    # ax2.hist(
    #     np.log10(masked_uncon),
    #     bins=nbins,
    #     histtype="step",
    #     lw=2,
    #     color="pink",
    #     density=True,
    #     label="Masked unconstr.",
    # )

    if field == "TT":
        ax2.axvline(
            np.log10(con),
            lw=2,
            color="dodgerblue",
            label="SMICA",
        )
        ax2.axvline(
            np.log10(masked_con),
            lw=2,
            color="forestgreen",
            label="Masked SMICA",
        )
    else:
        ax2.hist(
            np.log10(con),
            bins=nbins,
            histtype="step",
            lw=2,
            color="dodgerblue",
            density=True,
            label="Constr.",
        )
        ax2.hist(
            np.log10(masked_con),
            bins=nbins,
            histtype="step",
            lw=2,
            color="forestgreen",
            density=True,
            label="Masked constr.",
        )
        lower_con = np.percentile(np.log10(masked_con), 0.5)
        upper_con = np.percentile(np.log10(masked_con), 99.5)

        xlim = ax2.get_xlim()
        ax2.autoscale(enable=False)

        ax.set_xlim(10 ** np.array(ax2.get_xlim()))

        lower_axis = xlim[0]
        upper_axis = xlim[1]
        alpha = 0.2

        if concl:
            ax2.fill_between(
                x=np.linspace(lower_con, upper_con, num=100),
                y1=0,
                y2=10000 + 400,
                color="forestgreen",
                alpha=alpha,
            )

            # ax2.fill_between(
            #     x=np.linspace(lower_uncon, upper_con, num=100),
            #     y1=0,
            #     y2=10000 + 400,
            #     color="goldenrod",
            #     alpha=alpha,
            # )

            ax2.fill_between(
                x=np.linspace(lower_uncon, upper_uncon, num=100),
                y1=0,
                y2=10000 + 400,
                color="red",
                alpha=alpha,
            )

            ax2.fill_between(
                x=np.linspace(lower_axis, lower_con, num=100),
                y1=0,
                y2=10000 + 400,
                color="black",
                alpha=alpha / 2,
            )

            ax2.fill_between(
                x=np.linspace(upper_uncon, upper_axis, num=100),
                y1=0,
                y2=10000 + 400,
                color="black",
                alpha=alpha / 2,
            )

    ax.tick_params(
        direction="in", which="major", labelsize=14, width=1.5, length=5, zorder=101
    )
    ax.tick_params(
        direction="in", which="minor", labelsize=14, width=1.5, length=3, zorder=101
    )
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    [x.set_linewidth(1.5) for x in ax.spines.values()]

    ax.set_xlim(10 ** np.array(ax2.get_xlim()))
    ax2.set_xticklabels([])
    ax2.set_xlabel("")
    ax2.tick_params(direction="in", which="both", labelsize=12, width=0, zorder=101)

    ax.set_xlabel(
        rf"$\sum_{{\theta_{{\rm min}}, \theta_{{\rm max}}}}S^{{\rm {lab}}}_{{\theta_{{\rm min}}, \theta_{{\rm max}}}}$ {unit}",
        fontsize=17,
    )
    ax.set_ylabel("PDF", fontsize=17)
    ax2.legend(
        loc="best", frameon=True, framealpha=1, fancybox=True, ncol=1, fontsize=14
    )
    plt.tight_layout()
    if savefig:
        plt.savefig(
            fig_dir.joinpath("cumulative_" + field + f"_lmax{lmax}.png"), dpi=150
        )
    return


def save_signi(
    uncon: np.ndarray,
    con: np.ndarray,
    masked_con: np.ndarray,
    field: np.ndarray,
    signi_dict: dict,
    smica: float = 2,
    masked_smica: float = 3,
) -> None:
    """
    Save significance values from the data

    Parameters
    ----------
    uncon : np.ndarray
        Unconstrained data
    con : np.ndarray
        Constrained data
    masked_con : np.ndarray
        Masked constrained data
    field : np.ndarray
        The field
    signi_dict : dict
        The dictionary where to store significances above SMICA ones
    smica : float, optional
        SMICA value of significance
    masked_smica : float, optional
        Masked SMICA value of significance
    """

    uncon, _ = uncon
    con, _ = con
    masked_con, _ = masked_con
    smica, _ = smica
    masked_smica, _ = masked_smica
    if field == "TT":
        signi_dict[field] = [
            round(con, 2),
            round(masked_con, 2),
        ]
    else:
        signi_dict[field] = [
            round(np.mean(con > smica) * 100, 2),
            round(np.mean(masked_con > masked_smica) * 100, 2),
        ]

    return


def plot_signi(
    uncon: np.ndarray,
    con: np.ndarray,
    masked_con: np.ndarray,
    nbins: int,
    lmax: int,
    field: str,
    lab: str,
    fig_dir: str,
    *,
    smica: int = 2,
    masked_smica: int = 3,
    savefig: bool = False,
) -> None:
    """Plot significance of the unconstrained and the constrained (with and without masking) realizations.

    Parameters
    ----------
    uncon : np.ndarray
        unconstrained field
    con : np.ndarray
        constrained field
    masked_con : np.ndarray
        constrained field with masking
    nbins : int
        number of bins for the histogram
    lmax : int
        maximum spherical harmonic degree
    field : str
        name of the field
    lab : str
        label for the field
    fig_dir : str
        directory where to save the figure
    smica : int, optional
        smica value for the unconstrained field
    masked_smica : int, optional
        smica value for the constrained field with masking
    savefig : bool, optional
        if True, save the figure
    """

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    uncon, signi_uncon = uncon
    con, signi_con = con
    masked_con, signi_masked_con = masked_con
    smica, signi_smica = smica
    masked_smica, signi_masked_smica = masked_smica

    if field == "TT":
        ax.axvline(
            (con),
            lw=2,
            color="dodgerblue",
            label="SMICA",
        )
        ax.axvline(
            (masked_con),
            lw=2,
            color="forestgreen",
            label="Masked SMICA",
        )
    else:
        hist1, bins = np.histogram(con, bins=nbins, density=True)
        hist2, bins = np.histogram(masked_con, bins=nbins, density=True)
        ax.hist(
            (con),
            bins=nbins,
            histtype="step",
            lw=2,
            color="dodgerblue",
            density=True,
            label="Constr.",
        )
        ax.hist(
            (masked_con),
            bins=nbins,
            histtype="step",
            lw=2,
            color="forestgreen",
            density=True,
            label="Masked constr.",
        )

        ax.set_ylim(0, np.max([hist1, hist2]) + np.max([hist1, hist2]) * 0.5)

        ax.axvline(smica, lw=1.5, color="black", label="SMICA", ls=":")
        ax.axvline(masked_smica, lw=1.5, color="black", label="Masked SMICA", ls="--")

    ax.tick_params(
        direction="in", which="major", labelsize=14, width=1.5, length=5, zorder=101
    )
    ax.tick_params(
        direction="in", which="minor", labelsize=14, width=1.5, length=3, zorder=101
    )
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    [x.set_linewidth(1.5) for x in ax.spines.values()]

    ax.set_xlabel(rf"Significance with $S^{{\rm {lab}}}\ [\sigma]$", fontsize=15)
    ax.set_ylabel("PDF", fontsize=17)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=1,
        fancybox=True,
        ncol=1,
        fontsize=14,
    )
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath("signi_" + field + f"_lmax{lmax}.png"), dpi=150)
    if field == "TT":
        print(f"*--> Full-sky SMICA has a significance of {con} sigmas")
        print(f"*--> Masked SMICA has a significance of {masked_con} sigmas")
    else:
        count(con, obs=field, ref=smica)
        count(masked_con, obs="Masked " + field, ref=masked_smica)
    return


def plot_hist( # noqa: C901
    opt_angs: dict,
    field: str,
    lab: str,
    lmax: int,
    file_uncon: str,
    file_con: str,
    fig_dir: str,
    *,
    step: int = 1,
    nbins: int = 30,
    concl: bool = False,
    savefig: bool = False,
) -> None:
    """Plot histogram of the optimal angles.

    Parameters
    ----------
    opt_angs : dict
        optimal angles
    field : str
        name of the field
    lab : str
        label for the field
    lmax : int
        maximum spherical harmonic degree
    file_uncon : str
        path to the file containing the unconstrained realizations
    file_con : str
        path to the file containing the constrained realizations
    fig_dir : str
        directory where to save the figure
    step : int, optional
        step for the histogram
    nbins : int, optional
        number of bins for the histogram
    concl : bool, optional
        if True, plot the histogram of the constrained realization
    savefig : bool, optional
        if True, save the figure
    """

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    the_min = opt_angs[field][0]
    the_max = opt_angs[field][1]
    displ = opt_angs[field][2]

    start = int(180 / (lmax - 1))
    μ_min = np.arange(start, 180 + step, step)
    μ_max = np.arange(start, 180 + step, step)

    with open(file_uncon, "rb") as pickle_file:
        uncon = pickle.load(pickle_file)
    with open(file_con, "rb") as pickle_file:
        con = pickle.load(pickle_file)
    smica = False
    if len(con[0, 0]) == 1:
        smica = True
    mask = False
    if field[0] == "m":
        mask = True

    print(f"The optimal range is {the_min, the_max}!")
    print(f"The percentage displacement is {displ}!")

    idx_min = int(np.where(μ_min == the_min)[0])
    idx_max = int(np.where(μ_max == the_max)[0])

    ax.set(xscale="log", yscale="linear")
    ax2 = ax.twiny()

    hist1, bins = np.histogram(np.log10(uncon[idx_min, idx_max]), bins=nbins)
    if smica:
        hist, bins = np.histogram(np.log10(uncon[idx_min, idx_max]), bins=nbins)
    ax2.hist(
        np.log10(uncon[idx_min, idx_max]),
        bins=nbins,
        histtype="step",
        lw=2,
        color="red",
        density=False,
        label="Unconstrained",
    )
    lower_uncon = np.percentile(np.log10(uncon[idx_min, idx_max]), 1)
    upper_uncon = np.percentile(np.log10(uncon[idx_min, idx_max]), 99)
    uncon = 0

    color = "dodgerblue"
    if mask:
        color = "forestgreen"

    if smica:
        name = "SMICA"
        if mask:
            name = "Masked SMICA"
        ax2.axvline(
            x=np.log10(con[idx_min, idx_max]),
            lw=2,
            ls="-",
            color=color,
            label=name,
            zorder=-2,
        )
    else:
        name = "Constrained"
        if mask:
            name = "Masked constrained"
        hist2, bins = np.histogram(np.log10(con[idx_min, idx_max]), bins=nbins)
        ax2.hist(
            np.log10(con[idx_min, idx_max]),
            bins=nbins,
            histtype="step",
            lw=2,
            color=color,
            density=False,
            label=name,
        )
        lower_con = np.percentile(np.log10(con[idx_min, idx_max]), 1)
        upper_con = np.percentile(np.log10(con[idx_min, idx_max]), 99)
    con = 0

    xlim = ax2.get_xlim()
    ax2.autoscale(enable=False)

    lower_axis = xlim[0]
    upper_axis = xlim[1]
    alpha = 0.2

    if concl:
        ax2.fill_between(
            x=np.linspace(lower_con, upper_con, num=100),
            y1=0,
            y2=np.max([hist1, hist2]) + 400,
            color=color,
            alpha=alpha,
        )

        ax2.fill_between(
            x=np.linspace(lower_uncon, upper_uncon, num=100),
            y1=0,
            y2=np.max([hist1, hist2]) + 400,
            color="red",
            alpha=alpha,
        )

        ax2.fill_between(
            x=np.linspace(lower_axis, lower_con, num=100),
            y1=0,
            y2=np.max([hist1, hist2]) + 400,
            color="black",
            alpha=alpha,
        )

        ax2.fill_between(
            x=np.linspace(upper_uncon, upper_axis, num=100),
            y1=0,
            y2=np.max([hist1, hist2]) + 400,
            color="black",
            alpha=alpha,
        )

    if smica:
        ax.set_ylim(0, np.max(hist) + 400)
    else:
        ax.set_ylim(0, np.max([hist1, hist2]) + 400)
    ax.set_xlim(10 ** np.array(ax2.get_xlim()))

    ax.tick_params(
        direction="in", which="major", labelsize=14, width=1.5, length=5, zorder=101
    )
    ax.tick_params(
        direction="in", which="minor", labelsize=14, width=1.5, length=3, zorder=101
    )
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax2.set_xticklabels([])
    ax2.set_xlabel("")
    ax2.tick_params(direction="in", which="both", labelsize=12, width=0, zorder=101)
    [x.set_linewidth(1) for x in ax.spines.values()]

    ax.set_xlabel(
        rf"$S^{{{lab}}}_{{{the_min}^\circ, {the_max}^\circ}}\ [\mu K^4]$",
        fontsize=17,
    )
    ax.set_ylabel("Counts", fontsize=17)
    leg = ax2.legend(
        loc="best",
        frameon=True,
        framealpha=1,
        fancybox=True,
        ncol=1,
        fontsize=14,
    )
    leg.set_title(f"PD  = {displ}%", prop={"size": 14})
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath(f"{field}_hist_lmax{lmax}.png"), dpi=150)


def plot_joint_hist(
    opt_angs: dict,
    field: str,
    lab: str,
    lmax: int,
    file_uncon: str,
    file_con: str,
    fig_dir: str,
    *,
    concl: bool = False,
    step: int = 1,
    nbins: int = 30,
    savefig: bool = False,
) -> None:
    """
    Plot the joint histogram for a given field with the optimal angular
    ranges for the unconstrained and constrained case.

    Parameters
    ----------
    opt_angs : dict
        Dictionary with the optimal angular ranges for the unconstrained
        and constrained case.
    field : str
        Name of the field to be plotted.
    lab : str
        Name of the field in latex format.
    lmax : int
        Maximum multipole moment.
    file_uncon : str
        Path to the pickle file containing the unconstrained angular
        power spectrum.
    file_con : str
        Path to the pickle file containing the constrained angular
        power spectrum.
    fig_dir : str
        Path to the directory where the figure will be saved.
    concl : bool, optional
        If True, the confidence intervals will be plotted.
    step : int, optional
        Step size for the angular range.
    nbins : int, optional
        Number of bins for the histogram.
    savefig : bool, optional
        If True, the figure will be saved.
    """
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    the_min = opt_angs[field][0]
    the_max = opt_angs[field][1]
    displ = opt_angs[field][2]

    start = int(180 / (lmax - 1))
    μ_min = np.arange(start, 180 + step, step)
    μ_max = np.arange(start, 180 + step, step)

    with open(file_uncon, "rb") as pickle_file:
        uncon = pickle.load(pickle_file)
    with open(file_con, "rb") as pickle_file:
        con = pickle.load(pickle_file)

    mask = False
    if field[0] == "m":
        mask = True

    print(f"The optimal range is {the_min, the_max}!")
    print(f"The percentage displacement is {displ}!")

    idx_min = int(np.where(μ_min == the_min)[0])
    idx_max = int(np.where(μ_max == the_max)[0])

    ax.set(xscale="log", yscale="linear")
    ax2 = ax.twiny()

    hist1, bins = np.histogram(np.log10(uncon[idx_min, idx_max]), bins=nbins)
    hist2, bins = np.histogram(np.log10(con[idx_min, idx_max]), bins=nbins)

    ax2.hist(
        np.log10(uncon[idx_min, idx_max]),
        bins=nbins,
        histtype="step",
        lw=2,
        color="red",
        density=False,
        label="Unconstrained",
    )
    lower_uncon = np.percentile(np.log10(uncon[idx_min, idx_max]), 0.5)
    upper_uncon = np.percentile(np.log10(uncon[idx_min, idx_max]), 99.5)
    uncon = 0

    color = "dodgerblue"
    masked = ""
    if mask:
        color = "forestgreen"
        masked = "Masked"

    name = "Constrained"
    if mask:
        name = " constrained"

    ax2.hist(
        np.log10(con[idx_min, idx_max]),
        bins=nbins,
        histtype="step",
        lw=2,
        color=color,
        density=False,
        label=masked + name,
    )
    lower_con = np.percentile(np.log10(con[idx_min, idx_max]), 0.5)
    upper_con = np.percentile(np.log10(con[idx_min, idx_max]), 99.5)
    con = 0

    xlim = ax2.get_xlim()
    ax2.autoscale(enable=False)

    ax.set_ylim(0, np.max([hist1, hist2]) + 400)
    ax.set_xlim(10 ** np.array(ax2.get_xlim()))

    lower_axis = xlim[0]
    upper_axis = xlim[1]
    alpha = 0.2

    if concl:
        ax2.fill_between(
            x=np.linspace(lower_con, upper_con, num=100),
            y1=0,
            y2=np.max([hist1, hist2]) + 400,
            color=color,
            alpha=alpha,
        )

        # ax2.fill_between(
        #     x=np.linspace(lower_uncon, upper_con, num=100),
        #     y1=0,
        #     y2=np.max([hist1, hist2]) + 400,
        #     color="goldenrod",
        #     alpha=alpha,
        # )

        ax2.fill_between(
            x=np.linspace(lower_uncon, upper_uncon, num=100),
            y1=0,
            y2=np.max([hist1, hist2]) + 400,
            color="red",
            alpha=alpha,
        )

        ax2.fill_between(
            x=np.linspace(lower_axis, lower_con, num=100),
            y1=0,
            y2=np.max([hist1, hist2]) + 400,
            color="black",
            alpha=alpha / 2,
        )

        ax2.fill_between(
            x=np.linspace(upper_uncon, upper_axis, num=100),
            y1=0,
            y2=np.max([hist1, hist2]) + 400,
            color="black",
            alpha=alpha / 2,
        )

    ax.tick_params(
        direction="in", which="major", labelsize=14, width=1.5, length=5, zorder=101
    )
    ax.tick_params(
        direction="in", which="minor", labelsize=14, width=1.5, length=3, zorder=101
    )
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax2.set_xticklabels([])
    ax2.set_xlabel("")
    ax2.tick_params(direction="in", which="both", labelsize=12, width=0, zorder=101)
    [x.set_linewidth(1) for x in ax.spines.values()]

    ax.set_xlabel(
        rf"$S^{{{lab}}}_{{{the_min}^\circ, {the_max}^\circ}}$",
        fontsize=17,
    )
    ax.set_ylabel("Counts", fontsize=17)
    leg = ax2.legend(
        loc="best",
        frameon=True,
        framealpha=1,
        fancybox=True,
        ncol=1,
        fontsize=14,
    )
    leg.set_title(f"PD  = {displ}%", prop={"size": 14})
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath(f"{field}_hist_lmax{lmax}.png"), dpi=150)


def compute_ang_corr(
    uncon: np.ndarray,
    con: np.ndarray,
    CLs: dict,
    field: str,
    lab: str,
    P: np.ndarray,
    lmax: int,
    fig_dir: str,
    *,
    step: int = 1,
    savefig: bool = False,
) -> None:
    """
    Compute angular correlation functions and plots them.

    Parameters
    ----------
    uncon : np.ndarray
        Array of the unconstrained realizations.
    con : np.ndarray
        Array of the constrained realizations.
    CLs : dict
        Dictionary of the theoretical angular power spectra.
    field : str
        The field considered.
    lab : str
        The label of the field.
    P : np.ndarray
        The array of Legendre polynomials.
    lmax : int
        The maximum multipole of the angular correlation function.
    fig_dir : str
        The folder where to save the plots.
    step : int, optional
        The width of step in the angular domain.
    savefig : bool, optional
        Whether to save the plots.
    """

    ell = np.arange(0, lmax + 1, 1)
    start = int(180 / (lmax - 1))
    angs = np.arange(start, 180 + step, step)

    ang_uncon = (
        np.array((2 * ell + 1) * uncon[:, : lmax + 1]) @ np.array(P).T / (4 * np.pi)
    )

    ang_con = np.array((2 * ell + 1) * con) @ np.array(P).T / (4 * np.pi)

    ang_LCDM = (
        np.array((2 * ell + 1) * CLs[lab.lower()][: lmax + 1])
        @ np.array(P).T
        / (4 * np.pi)
    )

    uncon_16lev = []
    uncon_84lev = []
    uncon_2lev = []
    uncon_97lev = []

    con_16lev = []
    con_84lev = []
    con_2lev = []
    con_97lev = []

    for i in range(len(angs)):
        uncon_16lev.append(np.percentile(ang_uncon[:, i], 16))
        uncon_84lev.append(np.percentile(ang_uncon[:, i], 84))
        uncon_2lev.append(np.percentile(ang_uncon[:, i], 2.5))
        uncon_97lev.append(np.percentile(ang_uncon[:, i], 97.5))
        con_16lev.append(np.percentile(ang_con[:, i], 16))
        con_84lev.append(np.percentile(ang_con[:, i], 84))
        con_2lev.append(np.percentile(ang_con[:, i], 2.5))
        con_97lev.append(np.percentile(ang_con[:, i], 97.5))

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    plt.plot(
        angs,
        ang_LCDM,
        color="red",
        lw=2,
        label=r"$\Lambda$CDM",
        zorder=20,
    )

    color = "dodgerblue"
    name = "Constrained"
    if field[0] == "m":
        color = "forestgreen"
        name = "Masked constrained"

    plt.fill_between(
        angs,
        con_2lev,
        con_97lev,
        color=color,
        alpha=0.3,
        linewidth=0,
    )
    plt.fill_between(
        angs,
        con_16lev,
        con_84lev,
        color=color,
        alpha=0.7,
        linewidth=0,
    )

    plt.fill_between(
        angs,
        np.array(uncon_2lev),
        np.array(uncon_97lev),
        color="black",
        alpha=0.1,
        linewidth=0,
        zorder=-25,
    )
    plt.fill_between(
        angs,
        np.array(uncon_16lev),
        np.array(uncon_84lev),
        color="black",
        alpha=0.2,
        linewidth=0,
        zorder=-25,
    )

    plt.axhline(0, color="black", ls="--", lw=2, zorder=-20)

    plt.xlim([start, 180])

    plt.xlabel(r"$\theta\ [^\circ]$", fontsize=17)
    plt.ylabel(rf"$C^{{\rm {lab}}}(\theta)\ [\mu K^2]$", fontsize=17)
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(direction="in", which="both", labelsize=14, width=2, zorder=101)
    [x.set_linewidth(1.5) for x in ax.spines.values()]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="red",
            lw=2,
            label=r"$\Lambda$CDM",
            ls="-",
            markersize=0,
        ),
        Patch(
            facecolor="black",
            label=r"$\Lambda$CDM realizations",
            linewidth=0,
            alpha=0.2,
        ),
        Line2D(
            [0],
            [0],
            color=color,
            lw=2,
            label=name,
            ls="-",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        framealpha=1,
        fancybox=True,
        ncol=1,
        fontsize=14,
    )
    # plt.semilogy()
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath("ang_corr_" + field + f"_lmax{lmax}.png"), dpi=150)


def compute_ang_corr_TT(
    uncon: np.ndarray,
    con: np.ndarray,
    masked_con: np.ndarray,
    CLs: dict,
    P: np.ndarray,
    lmax: int,
    fig_dir: str,
    *,
    step: int = 1,
    savefig: bool = False,
) -> None:
    """
    Compute angular correlation functions for TT and plots them.

    Parameters
    ----------
    uncon : np.ndarray
        Array of the unconstrained realizations.
    con : np.ndarray
        Array of the constrained realizations.
    CLs : dict
        Dictionary of the theoretical angular power spectra.
    P : np.ndarray
        The array of Legendre polynomials.
    lmax : int
        The maximum multipole of the angular correlation function.
    fig_dir : str
        The folder where to save the plots.
    step : int, optional
        The width of step in the angular domain.
    savefig : bool, optional
        Whether to save the plots.
    """

    ell = np.arange(0, lmax + 1, 1)
    start = int(180 / (lmax - 1))
    angs = np.arange(start, 180 + step, step)

    ang_uncon = (
        np.array((2 * ell + 1) * uncon[:, : lmax + 1]) @ np.array(P).T / (4 * np.pi)
    )

    ang_con = np.array((2 * ell + 1) * con[0]) @ np.array(P).T / (4 * np.pi)
    ang_masked_con = (
        np.array((2 * ell + 1) * masked_con[0, 2:]) @ np.array(P).T / (4 * np.pi)
    )

    ang_LCDM = (
        np.array((2 * ell + 1) * CLs["tt"][: lmax + 1]) @ np.array(P).T / (4 * np.pi)
    )

    uncon_16lev = []
    uncon_84lev = []
    uncon_2lev = []
    uncon_97lev = []

    for i in range(len(angs)):
        uncon_16lev.append(np.percentile(ang_uncon[:, i], 16))
        uncon_84lev.append(np.percentile(ang_uncon[:, i], 84))
        uncon_2lev.append(np.percentile(ang_uncon[:, i], 2.5))
        uncon_97lev.append(np.percentile(ang_uncon[:, i], 97.5))

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    plt.plot(
        angs,
        ang_LCDM,
        color="red",
        lw=2,
        label=r"$\Lambda$CDM",
        zorder=20,
    )

    plt.plot(
        angs,
        ang_con,
        color="dodgerblue",
        lw=2,
        ls="--",
        label="SMICA",
        zorder=30,
    )

    plt.plot(
        angs,
        ang_masked_con,
        color="forestgreen",
        lw=2,
        ls="--",
        label=r"Masked SMICA",
        zorder=30,
    )

    plt.fill_between(
        angs,
        np.array(uncon_2lev),
        np.array(uncon_97lev),
        color="black",
        alpha=0.1,
        linewidth=0,
        zorder=-25,
    )

    plt.fill_between(
        angs,
        np.array(uncon_16lev),
        np.array(uncon_84lev),
        color="black",
        alpha=0.2,
        linewidth=0,
        zorder=-25,
        label=r"$\Lambda$CDM real.",
    )

    plt.axhline(0, color="black", ls="--", lw=2, zorder=-20)

    plt.xlim([start, 180])

    plt.xlabel(r"$\theta\ [^\circ]$", fontsize=17)
    plt.ylabel(r"$C^{TT}(\theta)\ [\mu K^2]$", fontsize=17)
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(direction="in", which="both", labelsize=14, width=2, zorder=101)
    [x.set_linewidth(1.5) for x in ax.spines.values()]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="red",
            lw=2,
            label=r"$\Lambda$CDM",
            ls="-",
            markersize=0,
        ),
        Patch(
            facecolor="black",
            label=r"$\Lambda$CDM realizations",
            linewidth=0,
            alpha=0.2,
        ),
        Line2D(
            [0],
            [0],
            color="dodgerblue",
            lw=2,
            label="SMICA",
            ls="-",
        ),
        Line2D(
            [0],
            [0],
            color="forestgreen",
            lw=2,
            label=r"Masked SMICA",
            ls="-",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        framealpha=1,
        fancybox=True,
        ncol=1,
        fontsize=14,
    )

    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath(f"ang_corr_TT_lmax{lmax}.png"), dpi=150)


def pvalue(uncon: np.ndarray, con: np.ndarray) -> float:
    """Compute the p-value for a given set of constrained and unconstraiend realizations.

    Parameters
    ----------
    uncon : np.ndarray
        Array of the unconstrained realizations.
    con : np.ndarray
        Array of the constrained realizations.

    Returns
    -------
    res : float
        The p-value.
    """

    res = np.zeros(len(uncon))
    if isinstance(con, np.float64):
        res = np.mean(uncon < con) * 100
        return res
    for i, val in enumerate(con):
        res[i] = np.mean(uncon < val) * 100
    res[res == 0] = 0.01
    return res


@ray.remote
def sigmas(uncon: np.ndarray, con: np.ndarray) -> tuple:
    """Compute the sigma values for a given constraint.

    Parameters
    ----------
    uncon : np.ndarray
        Array of the unconstrained realizations.
    con : np.ndarray
        Array of the constrained realizations.

    Returns
    -------
    res : tuple
        The sigma values.
    """
    # Convert the input values to log10 space.
    con = np.log10(con)
    uncon = np.log10(uncon)

    # Initialize the output values.
    res = np.zeros(len(uncon))

    # If there is a single constraint, compute the sigma value for that
    # constraint.
    if len(con) == 1:
        # Compute the fraction of unconstrained values that are less than the
        # constraint.
        res = np.mean(uncon < con)

        # Convert the fraction to a sigma value.
        res = norm.ppf(1 - res)
    else:
        # Otherwise, compute the sigma values for each constraint.
        for i, val in enumerate(con):
            res[i] = np.mean(uncon < val)

        # If a constraint is not satisfied, set its p-value to 1/N.
        res[res == 0] = 1 / len(uncon)

        # Compute the sigma values for each constraint.
        for i, val in enumerate(con):
            res[i] = norm.ppf(1 - res[i])

    # Compute the mean and standard deviation of the unconstrained values.
    mu, sigma = norm.fit(uncon)

    # If there is a single constraint, return the sigma value for that
    # constraint and the normalized distance between the constraint and the
    # mean of the unconstrained values.
    if len(con) == 1:
        return res, float(np.abs(mu - con) / sigma)

    # If there are multiple constraints, return the sigma values for each
    # constraint and the normalized distance between the mean of the
    # unconstrained values and the mean of the constraints.
    return res, np.abs(mu - con) / sigma


def save_sigmas(outfile: str, uncon: np.ndarray, con: np.ndarray) -> None:
    """Compute the sigma values and save them to a file.

    Parameters
    ----------
    outfile : str
        Path to the output file.
    uncon : np.ndarray
        Array of the unconstrained realizations.
    con : np.ndarray
        Array of the constrained realizations.
    """

    start = time.time()
    sigma = ray.get(sigmas.remote(read(uncon), read(con)))
    with open(outfile, "wb") as pickle_file:
        pickle.dump(sigma, pickle_file)
    end = time.time()
    print(f"Finished saving in {round(end-start, 2)} seconds!")
    return


# def plot_pvalue(
#     uncon,
#     con,
#     masked_con,
#     nbins,
#     lmax,
#     field,
#     lab,
#     fig_dir,
#     *,
#     smica=2,
#     masked_smica=3,
#     concl=False,
#     savefig=False,
# ):
#     plt.style.use("default")
#     fig, ax = plt.subplots(figsize=(7, 5))

#     ax.set(xscale="log", yscale="linear")
#     ax2 = ax.twiny()

#     hist1, bins = np.histogram(np.log10(masked_con), bins=nbins, density=True)
#     hist2, bins = np.histogram(np.log10(con), bins=nbins, density=True)

#     sigmas = [np.log10(4.550 / 2), np.log10(0.270 / 2), np.log10(0.006 / 2)]

#     # for i, val in enumerate(sigmas):
#     #     ax2.axvline(val, lw=1.5, color="goldenrod", ls=styles[i])
#     # ax2.text(val, 1.3, rf"{i+2}$\sigma$", color="goldenrod")

#     # ax.axvline(
#     #     np.log10(4.550 / 2), lw=1.5, color="goldenrod", label=r"2$\sigma$", ls="-"
#     # )
#     # ax.axvline(
#     #     np.log10(0.270 / 2), lw=1.5, color="goldenrod", label=r"3$\sigma$", ls="--"
#     # )
#     # ax.axvline(
#     #     np.log10(0.006 / 2), lw=1.5, color="goldenrod", label=r"4$\sigma$", ls=":"
#     # )

#     # ax2.hist(
#     #     np.log10(uncon),
#     #     bins=nbins,
#     #     histtype="step",
#     #     lw=2,
#     #     color="red",
#     #     density=True,
#     #     label="Unconstr.",
#     # )

#     if field == "TT":
#         ax2.axvline(
#             np.log10(con),
#             lw=2,
#             color="dodgerblue",
#             label="SMICA",
#         )
#         ax2.axvline(
#             np.log10(masked_con),
#             lw=2,
#             color="forestgreen",
#             label="Masked SMICA",
#         )
#     else:
#         ax2.hist(
#             np.log10(con),
#             bins=nbins,
#             histtype="step",
#             lw=2,
#             color="dodgerblue",
#             density=True,
#             label="Constr.",
#         )
#         ax2.hist(
#             np.log10(masked_con),
#             bins=nbins,
#             histtype="step",
#             lw=2,
#             color="forestgreen",
#             density=True,
#             label="Masked constr.",
#         )
#         ax2.axvline(np.log10(smica), lw=1.5, c="k", label="SMICA", ls=":")
#         ax2.axvline(
#             np.log10(masked_smica), lw=1.5, c="k", label="Masked SMICA", ls="--"
#         )

#     ax2.autoscale(enable=False)

#     ax.set_ylim(0, np.max([hist1, hist2]) + 0.7)
#     ax.set_xlim(10 ** np.array(ax2.get_xlim()))

#     color = "black"
#     alpha = 0.07
#     low = np.log10(0.001)
#     if concl:
#         ax2.fill_between(
#             x=np.linspace(low, np.log10(31.73 / 2), num=100),
#             y1=0,
#             y2=np.max([hist1, hist2]) + 1,
#             color=color,
#             alpha=alpha,
#         )

#         ax2.fill_between(
#             x=np.linspace(low, sigmas[0], num=100),
#             y1=0,
#             y2=np.max([hist1, hist2]) + 1,
#             color=color,
#             alpha=alpha,
#         )

#         ax2.fill_between(
#             x=np.linspace(low, sigmas[1], num=100),
#             y1=0,
#             y2=np.max([hist1, hist2]) + 1,
#             color=color,
#             alpha=alpha,
#         )

#         ax2.fill_between(
#             x=np.linspace(low, sigmas[2], num=100),
#             y1=0,
#             y2=np.max([hist1, hist2]) + 1,
#             color=color,
#             alpha=alpha,
#         )

#     # func.plot_gauss(
#     #     (uncon),
#     #     (masked_con),
#     #     label="Gaussian fit",
#     #     color="darkorange",
#     #     ls="--",
#     #     verbose=False,
#     # )
#     ax.tick_params(
#         direction="in", which="major", labelsize=14, width=1.5, length=5, zorder=101
#     )
#     ax.tick_params(
#         direction="in", which="minor", labelsize=14, width=1.5, length=3, zorder=101
#     )
#     ax.yaxis.set_ticks_position("both")
#     ax.xaxis.set_ticks_position("both")
#     [x.set_linewidth(1.5) for x in ax.spines.values()]

#     # ax.set_ylim(0, np.max([hist1, hist2]) + 1)
#     ax2.set_xticklabels([])
#     ax2.set_xlabel("")
#     ax2.tick_params(direction="in", which="both", labelsize=12, width=0, zorder=101)

#     ax.set_xlabel(rf"p-value with $S^{{\rm {lab}}}\ [\%]$", fontsize=15)
#     ax.set_ylabel("PDF", fontsize=17)
#     ax2.legend(
#         loc="best", frameon=True, framealpha=1, fancybox=True, ncol=1, fontsize=14
#     )
#     plt.tight_layout()
#     if savefig:
#         plt.savefig(fig_dir.joinpath("pvalue_" + field + f"_lmax{lmax}.png"), dpi=150)
#     if field == "TT":
#         print(f"*--> Full-sky SMICA has a p-value of {round(con, 2)} %")
#         print(f"*--> Masked SMICA has a p-value of {round(masked_con, 2)} %")
#     else:
#         count_down(np.log10(con), obs=field, ref=np.log10(smica))
#         count_down(
#             np.log10(masked_con), obs="Masked " + field, ref=np.log10(masked_smica)
#         )

#     return


def count_down(array: np.ndarray, obs: str = "OBSERVABLE", ref: float = None) -> None:
    """
    Count the number of elements in an array that are below a reference value.

    Parameters
    ----------
    array : ndarray
        Array to be checked.
    obs : str, optional
        Name of the observable.
    ref : float, optional
        Reference value. If None, 0.1% is used. Default: None.
    """

    if ref is None:
        print(
            rf"*--> For {obs}, the {round(np.mean(array < 0.1) * 100, 2)}% is below 0.1%"
        )
    else:
        if ref < 0:
            print(
                rf"*--> For {obs}, the {round(np.mean(array < ref) * 100, 2)}% is below Masked SMICA ({round(10**ref, 2)}%)"
            )
        else:
            print(
                rf"*--> For {obs}, the {round(np.mean(array < ref) * 100, 2)}% is below SMICA ({round(10**ref, 2)}%)"
            )

    lower = round(10 ** (np.percentile(array, 2.5)), 2)
    upper = round(10 ** (np.percentile(array, 97.5)), 2)
    print(rf"*--> For {obs}, the mean is {round(10 ** (np.mean(array)), 2)}")
    print(rf"*--> For {obs}, the 95% CL is [{lower}, {upper}]")

    return


def downgrade_map(
    inmap: np.ndarray, NSIDEout: int, fwhmout: float = None
) -> np.ndarray:
    """Downgrade a healpix map to a lower resolution map

    Parameters
    ----------
    inmap : np.ndarray
        healpix map to be downgraded
    NSIDEout : int
        NSIDE of output map
    fwhmout : float, optional
        fwhm of output map in degrees, if None, then use NSIDE to calculate fwhm

    Returns
    -------
    outmap : np.ndarray
        downgraded map
    """

    # get coefficent to covolve with beam and pixel window func
    plout = hp.pixwin(NSIDEout)
    lmax = plout.size - 1
    # print "LMAX is ",lmax
    NSIDEin = hp.get_nside(inmap)
    plin = hp.pixwin(NSIDEin)[: lmax + 1]
    fwhmin = hp.nside2resol(NSIDEin, arcmin=False)
    blin = hp.gauss_beam(fwhmin, lmax=lmax)
    if fwhmout is None:
        fwhmout = hp.nside2resol(NSIDEout, arcmin=False)
    blout = hp.gauss_beam(fwhmout, lmax=lmax)
    multby = blout * plout / (blin * plin)  # one number per ell

    # turn map to spherical harmonics, colvolve, then turn back into map
    alm = hp.map2alm(inmap, lmax)
    alm = hp.almxfl(alm, multby)  # colvolve w/window funcs
    return hp.alm2map(alm, NSIDEout, verbose=False)
