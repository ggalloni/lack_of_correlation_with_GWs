import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import ray
from progressbar import ProgressBar

from GW_lack_corr.classes.state import State


def get_random_seeds(CurrentState: State) -> np.ndarray:
    return CurrentState.settings.get_seeds()


def produce_realizations(CurrentState: State) -> None:
    """Produce the realizations.

    Parameters
    ----------
    CurrentState : State
        The current state object.

    """
    # Get number of realizations
    N = min(1000, CurrentState.settings.N)

    # Get the seeds
    seeds = get_random_seeds(CurrentState)
    free_TT_seeds = seeds["TT_uncon"]

    # Initialize empty dictionaries for the realizations
    free_TT_maps: list = []

    # Get the current power spectra
    CLs: dict = CurrentState.CLs
    lmax: int = CurrentState.settings.lmax
    nside: int = CurrentState.settings.nside

    widgets = CurrentState.widgets
    widgets[0] = "Generating realizations: "
    pbar = ProgressBar(widgets=widgets, maxval=N).start()
    start = time.time()
    for i in pbar(i for i in range(N)):
        np.random.seed(free_TT_seeds[i])  # noqa: NPY002
        free_TT_maps.append(hp.synfast(CLs["tt"], nside=nside, lmax=lmax))
    end = time.time()
    pbar.finish()

    print(f"Done in {round(end-start, 2)} seconds!")

    return free_TT_maps


def masking(mapp, mask):
    mask = np.logical_not(mask.astype(bool))
    masked = mapp.copy()
    masked[mask] = hp.UNSEEN
    return masked


@ray.remote
def mask_maps(N, free_TT_maps, mask, lmax, nside, inv_window):
    masked_TT_maps = []
    reconstructed_TT_maps = []

    for i in range(N):
        masked_TT_maps.append(masking(free_TT_maps[i], mask))
        alm = hp.map2alm(masked_TT_maps[i], lmax=lmax, use_pixel_weights=True)
        map_recon = hp.alm2map(alm @ inv_window, nside=nside, pol=False)
        reconstructed_TT_maps.append(map_recon)
    return masked_TT_maps, reconstructed_TT_maps


def get_masked_maps(
    CurrentState: State,
    free_TT_maps: list,
) -> tuple:
    """
    Compute the masked angular spectra of the free and constrained realizations
    of the temperature and gravitational wave fields.
    """
    N = min(1000, CurrentState.settings.N)

    inv_window = np.linalg.inv(CurrentState.get_window())

    free_TT_maps_id = ray.put(free_TT_maps)
    mask_id = ray.put(CurrentState.mask)
    lmax_id = ray.put(CurrentState.settings.lmax)
    nside_id = ray.put(CurrentState.settings.nside)
    inv_window_id = ray.put(inv_window)
    N_id = ray.put(N)

    start = time.time()
    masked_TT_maps, reconstructed_TT_maps = ray.get(
        mask_maps.remote(
            N_id, free_TT_maps_id, mask_id, lmax_id, nside_id, inv_window_id
        )
    )
    end = time.time()
    print(f"Done masked and reconstructed maps in {round(end-start, 2)} seconds!")

    return masked_TT_maps, reconstructed_TT_maps


def compute_variances(
    CurrentState: State,
    free_TT_maps: list,
    masked_TT_maps: list,
    reconstructed_TT_maps: list,
):
    variance_free_TT = np.sqrt(
        np.var(
            free_TT_maps,
            axis=0,
        )
    )

    variance_masked_TT = np.sqrt(np.var(masked_TT_maps, axis=0))
    mask = CurrentState.mask
    variance_masked_TT = masking(variance_masked_TT, mask)

    variance_reconstructed_TT = np.sqrt(np.var(reconstructed_TT_maps, axis=0))

    return variance_free_TT, variance_masked_TT, variance_reconstructed_TT


def main(CurrentState: State):
    free_TT_maps = produce_realizations(CurrentState)
    masked_TT_maps, reconstructed_TT_maps = get_masked_maps(CurrentState, free_TT_maps)
    variance_free_TT, variance_masked_TT, variance_reconstructed_TT = compute_variances(
        CurrentState, free_TT_maps, masked_TT_maps, reconstructed_TT_maps
    )

    dpi: int = 150
    figsize_inch: tuple = 7, 3.5
    cmap: str = CurrentState.settings.cmap

    savefig = CurrentState.settings.savefig
    show = CurrentState.settings.show

    # Initialize a figure to plot the sky map on
    fig: plt.Figure = plt.figure(figsize=figsize_inch, dpi=dpi)

    # Plot the sky map
    hp.mollview(
        variance_free_TT,
        fig=fig.number,
        xsize=figsize_inch[0] * dpi,
        title="",
        cbar=True,
        cmap=cmap,
        unit=r"$\sigma_{\rm pix} [\mu K]$",
        # min=-scale,
        # max=scale,
        bgcolor="none",
    )
    hp.graticule(dmer=360, dpar=360, alpha=0)
    if savefig:
        plt.savefig(
            CurrentState.settings.fig_dir.joinpath(
                CurrentState.settings.prefix + "variance_fullsky_TT.png"
            )
        )

    fig: plt.Figure = plt.figure(figsize=figsize_inch, dpi=dpi)
    hp.mollview(
        variance_masked_TT,
        fig=fig.number,
        xsize=figsize_inch[0] * dpi,
        title="",
        cbar=True,
        cmap=cmap,
        unit=r"$\sigma_{\rm pix} [\mu K]$",
        # min=-scale,
        # max=scale,
        bgcolor="none",
    )
    hp.graticule(dmer=360, dpar=360, alpha=0)

    fig: plt.Figure = plt.figure(figsize=figsize_inch, dpi=dpi)
    hp.mollview(
        variance_reconstructed_TT,
        fig=fig.number,
        xsize=figsize_inch[0] * dpi,
        title="",
        cbar=True,
        cmap=cmap,
        unit=r"$\sigma_{\rm pix} [\mu K]$",
        # min=-scale,
        # max=scale,
        bgcolor="none",
    )
    hp.graticule(dmer=360, dpar=360, alpha=0)
    if savefig:
        plt.savefig(
            CurrentState.settings.fig_dir.joinpath(
                CurrentState.settings.prefix + "variance_reconstructed_TT.png"
            )
        )

    fig: plt.Figure = plt.figure(figsize=figsize_inch, dpi=dpi)
    hp.mollview(
        variance_reconstructed_TT - variance_free_TT,
        fig=fig.number,
        xsize=figsize_inch[0] * dpi,
        title="",
        cbar=True,
        cmap=cmap,
        unit=r"$\sigma_{\rm pix} [\mu K]$",
        # min=-scale,
        # max=scale,
        bgcolor="none",
    )
    hp.graticule(dmer=360, dpar=360, alpha=0)
    if savefig:
        plt.savefig(
            CurrentState.settings.fig_dir.joinpath(
                CurrentState.settings.prefix + "variance_difference.png"
            )
        )

    fig: plt.Figure = plt.figure(figsize=figsize_inch, dpi=dpi)
    hp.mollview(
        masking(variance_reconstructed_TT - variance_free_TT, CurrentState.mask),
        fig=fig.number,
        xsize=figsize_inch[0] * dpi,
        title="",
        cbar=True,
        cmap=cmap,
        unit=r"$\sigma_{\rm pix} [\mu K]$",
        # min=-scale,
        # max=scale,
        bgcolor="none",
    )
    hp.graticule(dmer=360, dpar=360, alpha=0)
    if savefig:
        plt.savefig(
            CurrentState.settings.fig_dir.joinpath(
                CurrentState.settings.prefix + "variance_difference_masked.png"
            )
        )

    if show:
        plt.show()

    plt.close("all")
    return


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------ #
