import pickle
import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import ray
from progressbar import ProgressBar

from functions import mask_TT_un, mask_X_con, mask_X_un


def get_SMICA_seeds(CurrentState):
    lmax = CurrentState.settings.lmax
    cmb_map = CurrentState.cmb_map
    mask = CurrentState.mask
    CLs = CurrentState.CLs

    fullsky_SMICA_alm = hp.map2alm(cmb_map, lmax=lmax, use_pixel_weights=True)
    f_T = nmt.NmtField(mask, [cmb_map])
    masked_SMICA_alm = f_T.get_alms()[0]

    filt = np.ones(lmax + 1)
    masked_SMICA_alm = hp.almxfl(masked_SMICA_alm, filt)
    masked_SMICA_alm = masked_SMICA_alm[masked_SMICA_alm != 0]

    inv_window = np.linalg.inv(CurrentState.get_window())
    decoupled_SMICA_alm = masked_SMICA_alm @ inv_window
    decoupled_SMICA_seed = hp.almxfl(
        decoupled_SMICA_alm, CLs["tgw"][: lmax + 1] / CLs["tt"][: lmax + 1]
    )
    decoupled_SMICA_seed[np.isnan(decoupled_SMICA_seed)] = 0
    decoupled_SMICA_seed[np.isinf(decoupled_SMICA_seed)] = 0

    SMICA_seed = hp.almxfl(
        fullsky_SMICA_alm, CLs["tgw"][: lmax + 1] / CLs["tt"][: lmax + 1]
    )
    SMICA_seed[np.isnan(SMICA_seed)] = 0

    return SMICA_seed, decoupled_SMICA_seed, fullsky_SMICA_alm, masked_SMICA_alm

    # -------------------------------------------------------------------------------- #


def plot_sky(CurrentState, sky_map, unit, scale, name):
    dpi = 150
    figsize_inch = 7, 3.5
    cmap = CurrentState.settings.cmap

    fig = plt.figure(figsize=figsize_inch, dpi=dpi)
    hp.mollview(
        sky_map,
        fig=fig.number,
        xsize=figsize_inch[0] * dpi,
        title="",
        cbar=True,
        cmap=cmap,
        unit=unit,
        min=-scale,
        max=scale,
        bgcolor="none",
    )
    hp.graticule(dmer=360, dpar=360, alpha=0)
    if CurrentState.settings.savefig:
        plt.savefig(
            CurrentState.settings.dir.joinpath(
                "../figures/" + CurrentState.settings.prefix + name
            )
        )
    return


def produce_realizations(CurrentState):
    N = CurrentState.settings.N

    free_GWGW_realizations = {}
    free_TT_realizations = {}
    LCDM_constrained_GWGW_realizations = {}
    fullsky_constrained_GWGW_realizations = {}
    masked_constrained_GWGW_realizations = {}

    CLs = CurrentState.CLs
    lmax = CurrentState.settings.lmax
    SMICA_seed, decoupled_SMICA_seed, fullsky_SMICA_alm, _ = get_SMICA_seeds(
        CurrentState
    )

    widgets = CurrentState.widgets
    widgets[0] = "Generating realizations: "
    pbar = ProgressBar(widgets=widgets, maxval=N).start()
    start = time.time()
    for i in pbar(i for i in range(N)):
        free_TT_realizations["%s" % i] = hp.synalm(CLs["tt"], lmax=lmax)
        free_GWGW_realizations["%s" % i] = hp.synalm(CLs["gwgw"], lmax=lmax)
        masked_constrained_GWGW_realizations["%s" % i] = (
            hp.synalm(CLs["constrained_gwgw"], lmax=lmax) + decoupled_SMICA_seed
        )
        fullsky_constrained_GWGW_realizations["%s" % i] = (
            hp.synalm(CLs["constrained_gwgw"], lmax=lmax) + SMICA_seed
        )
        seed = hp.almxfl(
            free_TT_realizations["%s" % i],
            CLs["tgw"][: lmax + 1] / CLs["tt"][: lmax + 1],
        )
        seed[np.isnan(seed)] = 0
        LCDM_constrained_GWGW_realizations["%s" % i] = (
            hp.synalm(CLs["constrained_gwgw"], lmax=lmax) + seed
        )
    end = time.time()
    pbar.finish()

    print(f"Done in {round(end-start, 2)} seconds!")

    if CurrentState.settings.debug:
        nside = CurrentState.settings.nside
        plot_sky(
            CurrentState,
            hp.alm2map(fullsky_SMICA_alm, nside=nside),
            unit=r"$\Delta T\ [\mu K]$",
            scale=100,
            name="planck_map.png",
        )
        masked_CMB_sky = hp.ma(hp.alm2map(fullsky_SMICA_alm, nside=nside))
        masked_CMB_sky.mask = np.logical_not(CurrentState.mask)
        plot_sky(
            CurrentState,
            masked_CMB_sky,
            unit=r"$\Delta T\ [\mu K]$",
            scale=100,
            name="masked_planck_map.png",
        )
        plot_sky(
            CurrentState,
            hp.alm2map(fullsky_constrained_GWGW_realizations["0"], nside=nside),
            unit=r"$\delta_{GW}\times T_0\ [\mu K]$",
            scale=250,
            name="con_map.png",
        )
        plot_sky(
            CurrentState,
            hp.alm2map(masked_constrained_GWGW_realizations["0"], nside=nside),
            unit=r"$\delta_{GW}\times T_0\ [\mu K]$",
            scale=250,
            name="con_decoupled_map.png",
        )
        if CurrentState.settings.show:
            plt.show()

    return (
        free_TT_realizations,
        free_GWGW_realizations,
        LCDM_constrained_GWGW_realizations,
        fullsky_constrained_GWGW_realizations,
        masked_constrained_GWGW_realizations,
    )


# ------------------------------------------------------------------------------------ #


def compute_fullsky_angular_spectra(
    CurrentState,
    free_TT_realizations,
    free_GWGW_realizations,
    LCDM_constrained_GWGW_realizations,
    fullsky_constrained_GWGW_realizations,
    masked_constrained_GWGW_realizations,
):
    N = CurrentState.settings.N
    lmax = CurrentState.settings.lmax
    _, _, fullsky_SMICA_alm, _ = get_SMICA_seeds(CurrentState)
    GWGW_uncon_CL = []
    TT_uncon_CL = []
    GWGW_con_CL = []
    masked_GWGW_con_CL = []
    TGW_uncon_CL = []
    TGW_con_CL = []

    widgets = CurrentState.widgets
    widgets[0] = "Full-sky spectra: "
    pbar = ProgressBar(widgets=widgets, maxval=N).start()
    start = time.time()
    for i in pbar(i for i in range(N)):
        TT_uncon_CL.append(hp.alm2cl(free_TT_realizations["%s" % i], lmax=lmax))
        GWGW_uncon_CL.append(hp.alm2cl(free_GWGW_realizations["%s" % i], lmax=lmax))
        TGW_uncon_CL.append(
            hp.alm2cl(
                free_TT_realizations["%s" % i],
                LCDM_constrained_GWGW_realizations["%s" % i],
                lmax=lmax,
            )
        )
        masked_GWGW_con_CL.append(
            hp.alm2cl(masked_constrained_GWGW_realizations["%s" % i], lmax=lmax)
        )
        GWGW_con_CL.append(
            hp.alm2cl(fullsky_constrained_GWGW_realizations["%s" % i], lmax=lmax)
        )
        TGW_con_CL.append(
            hp.alm2cl(
                fullsky_SMICA_alm,
                fullsky_constrained_GWGW_realizations["%s" % i],
                lmax=lmax,
            )
        )
    end = time.time()
    pbar.finish()
    print(f"Done in {round(end-start, 2)} seconds!")

    if CurrentState.settings.debug:
        ell = np.arange(2, lmax + 1, 1)
        plt.plot(ell, np.mean(TGW_uncon_CL, axis=0)[2:], label="Uncon")
        plt.plot(ell, np.mean(TGW_con_CL, axis=0)[2:], label="Con")
        plt.plot(ell, CurrentState.CLs["tgw"][2 : lmax + 1], label="Theo", ls="--")
        plt.plot(ell, hp.alm2cl(fullsky_SMICA_alm, lmax=lmax)[2:], label="Planck")

        plt.legend()
        plt.show()
    return (
        TT_uncon_CL,
        GWGW_uncon_CL,
        GWGW_con_CL,
        masked_GWGW_con_CL,
        TGW_uncon_CL,
        TGW_con_CL,
    )


# ------------------------------------------------------------------------------------ #


def compute_masked_angular_spectra(
    CurrentState,
    free_TT_realizations,
    masked_constrained_GWGW_realizations,
):
    N = CurrentState.settings.N

    cmb_map_id = ray.put(CurrentState.cmb_map)
    cmb_real_id = ray.put(free_TT_realizations)
    cgwb_masked_con_real_id = ray.put(masked_constrained_GWGW_realizations)
    mask_id = ray.put(CurrentState.mask)
    lmax_id = ray.put(CurrentState.settings.lmax)
    nside_id = ray.put(CurrentState.settings.nside)
    inv_window_id = ray.put(np.linalg.inv(CurrentState.get_window()))
    CLs_id = ray.put(CurrentState.CLs)
    N_id = ray.put(N)

    start = time.time()
    masked_TGW_uncon_CL = mask_X_un.remote(
        N_id, cmb_real_id, mask_id, lmax_id, nside_id, inv_window_id, CLs_id
    )
    masked_TGW_uncon_CL = ray.get(masked_TGW_uncon_CL)
    end = time.time()
    print(f"Done masked TGW unconstrained spectra in {round(end-start, 2)} seconds!")

    start = time.time()
    masked_TGW_con_CL = mask_X_con.remote(
        N_id, cmb_map_id, cgwb_masked_con_real_id, mask_id, lmax_id, nside_id
    )
    masked_TGW_con_CL = ray.get(masked_TGW_con_CL)
    end = time.time()
    print(f"Done masked TGW constrained spectra in {round(end-start, 2)} seconds!")

    start = time.time()
    masked_TT_uncon_CL = mask_TT_un.remote(
        N_id, cmb_real_id, mask_id, lmax_id, nside_id
    )
    masked_TT_uncon_CL = ray.get(masked_TT_uncon_CL)
    end = time.time()
    print(f"Done masked TT unconstrained spectra in {round(end-start, 2)} seconds!")
    ray.shutdown()

    if CurrentState.settings.debug:
        ell = np.arange(2, CurrentState.settings.lmax + 1, 1)
        plt.plot(ell, np.mean(masked_TGW_uncon_CL, axis=0)[2:], label="Uncon")
        plt.plot(ell, np.mean(masked_TGW_con_CL, axis=0)[2:], label="Con")
        plt.plot(
            ell,
            CurrentState.CLs["tgw"][2 : CurrentState.settings.lmax + 1],
            label="Theo",
            ls="--",
        )

        plt.semilogy()

        plt.legend()
        plt.show()
    return masked_TGW_uncon_CL, masked_TGW_con_CL, masked_TT_uncon_CL


def saving_spectra(
    CurrentState,
    TT_uncon_CL,
    GWGW_uncon_CL,
    GWGW_con_CL,
    masked_GWGW_con_CL,
    TGW_uncon_CL,
    TGW_con_CL,
    masked_TGW_uncon_CL,
    masked_TGW_con_CL,
    masked_TT_uncon_CL,
):
    new_data = [
        TT_uncon_CL,
        GWGW_uncon_CL,
        masked_TT_uncon_CL,
        GWGW_con_CL,
        masked_GWGW_con_CL,
        TGW_uncon_CL,
        TGW_con_CL,
        masked_TGW_uncon_CL,
        masked_TGW_con_CL,
    ]

    if CurrentState.settings.debug:
        exit()

    files = CurrentState.settings.get_spectra_files()
    for dat, file in zip(new_data, files):
        with open(file, "wb") as pickle_file:
            pickle.dump(np.asarray(dat), pickle_file)
    new_data = 0
    return


def main(CurrentState):
    (
        free_TT_realizations,
        free_GWGW_realizations,
        LCDM_constrained_GWGW_realizations,
        fullsky_constrained_GWGW_realizations,
        masked_constrained_GWGW_realizations,
    ) = produce_realizations(CurrentState)

    (
        TT_uncon_CL,
        GWGW_uncon_CL,
        GWGW_con_CL,
        masked_GWGW_con_CL,
        TGW_uncon_CL,
        TGW_con_CL,
    ) = compute_fullsky_angular_spectra(
        CurrentState,
        free_TT_realizations,
        free_GWGW_realizations,
        LCDM_constrained_GWGW_realizations,
        fullsky_constrained_GWGW_realizations,
        masked_constrained_GWGW_realizations,
    )
    (
        masked_TGW_uncon_CL,
        masked_TGW_con_CL,
        masked_TT_uncon_CL,
    ) = compute_masked_angular_spectra(
        CurrentState,
        free_TT_realizations,
        masked_constrained_GWGW_realizations,
    )
    saving_spectra(
        CurrentState,
        TT_uncon_CL,
        GWGW_uncon_CL,
        GWGW_con_CL,
        masked_GWGW_con_CL,
        TGW_uncon_CL,
        TGW_con_CL,
        masked_TGW_uncon_CL,
        masked_TGW_con_CL,
        masked_TT_uncon_CL,
    )

    return


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------ #
