import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from functions import (
    integrate_spherical_harmonics,
    integrate_spherical_harmonics_pix,
    integrate_spherical_harmonics_pix_mask,
    build_window_function,
)


def test_Ylm_intergral(CurrentState):
    l1, m1 = 2, 1
    l2, m2 = 2, 1
    nside = CurrentState.settings.nside

    integral, expected = integrate_spherical_harmonics(l1, m1, l2, m2)

    print(f"Computed integral: {integral}")
    print(f"Expected value: {expected}")

    integral, expected = integrate_spherical_harmonics_pix(l1, m1, l2, m2, nside)

    print(f"Computed integral: {integral}")
    print(f"Expected value: {expected}")

    integral, expected = integrate_spherical_harmonics_pix_mask(
        l1, m1, l2, m2, nside, CurrentState.mask
    )

    print(f"Computed integral: {integral}")
    print(f"Expected value: {expected}")
    return


def plot_window_function(window_function, CurrentState):
    lmax = CurrentState.settings.lmax
    alm_size = hp.Alm.getsize(lmax)
    sizes = [lmax + 1 - m for m in range(lmax + 1)]
    cumul_sizes = np.cumsum(sizes)
    diff_sizes = np.diff(cumul_sizes)
    ticks = np.array(
        [cumul_sizes[i] - diff_sizes[i] / 2 for i in range(len(diff_sizes))]
    )

    normalized_window = window_function.real.copy()
    for i, val in enumerate(np.diag(window_function.real)):
        normalized_window[i, :] /= np.sqrt(val)
        normalized_window[:, i] /= np.sqrt(val)

    normalized_window[normalized_window == np.nan] = 0
    normalized_window[normalized_window == np.inf] = 0

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.pcolormesh(
        range(alm_size),
        np.flip(range(alm_size)),
        normalized_window,
        shading="nearest",
        cmap=CurrentState.settings.cmap,
    )

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(
        r"Normalized $W_{\ell m}^{\ell' m'}$",
        fontsize=15,
    )
    ax.tick_params(
        direction="in", which="major", labelsize=14, width=1.5, length=5, zorder=101
    )
    ax.tick_params(
        direction="in", which="minor", labelsize=14, width=1.5, length=3, zorder=101
    )
    ax.set_xticks(ticks=ticks - 1, labels=range(lmax))
    ax.set_yticks(ticks=np.abs(alm_size - ticks), labels=range(lmax))
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    [x.set_linewidth(1.5) for x in ax.spines.values()]

    ax.set_xlabel(r"$m$", fontsize=17)
    ax.set_ylabel(r"$m^\prime$", fontsize=17)
    plt.tight_layout()
    if CurrentState.settings.savefig:
        plt.savefig(
            CurrentState.settings.dir.joinpath(f"../figures/{lmax}_window.png"),
            dpi=150,
        )
    return


def plot_cutsky_covariance(window_function, CurrentState):
    lmax = CurrentState.settings.lmax
    CLs = CurrentState.CLs
    alm_size = hp.Alm.getsize(lmax)
    sizes = [lmax + 1 - m for m in range(lmax + 1)]
    cumul_sizes = np.cumsum(sizes)
    diff_sizes = np.diff(cumul_sizes)
    ticks = np.array(
        [cumul_sizes[i] - diff_sizes[i] / 2 for i in range(len(diff_sizes))]
    )

    fullsky_COV = np.ones(alm_size)
    fullsky_COV = np.real(hp.almxfl(fullsky_COV, CLs["tt"][: lmax + 1]))
    fullsky_COV = np.diag(fullsky_COV)

    COV = window_function @ fullsky_COV @ window_function.conj()

    normalized_COV = COV.real.copy()
    for i, val in enumerate(np.diag(COV.real)):
        normalized_COV[i, :] /= np.sqrt(val)
        normalized_COV[:, i] /= np.sqrt(val)

    normalized_COV[normalized_COV == np.nan] = 0
    normalized_COV[normalized_COV == np.inf] = 0

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.pcolormesh(
        range(alm_size),
        np.flip(range(alm_size)),
        normalized_COV,
        shading="nearest",
        cmap=CurrentState.settings.cmap,
    )

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(
        r"Normalized $Cov^{CS}$",
        fontsize=15,
    )
    ax.tick_params(
        direction="in", which="major", labelsize=14, width=1.5, length=5, zorder=101
    )
    ax.tick_params(
        direction="in", which="minor", labelsize=14, width=1.5, length=3, zorder=101
    )
    ax.set_xticks(ticks=ticks - 1, labels=range(lmax))
    ax.set_yticks(ticks=np.abs(alm_size - ticks), labels=range(lmax))
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    [x.set_linewidth(1.5) for x in ax.spines.values()]

    ax.set_xlabel(r"$m$", fontsize=17)
    ax.set_ylabel(r"$m^\prime$", fontsize=17)
    plt.tight_layout()
    if CurrentState.settings.savefig:
        plt.savefig(
            CurrentState.settings.dir.joinpath(f"../figures/{lmax}_COV_CS.png"), dpi=150
        )
    return


def get_window_function(CurrentState):
    if CurrentState.settings.debug:
        test_Ylm_intergral(CurrentState)

    window_function = build_window_function(
        CurrentState.settings.lmax, CurrentState.settings.nside, CurrentState.mask
    )

    if CurrentState.settings.debug:
        plot_window_function(window_function, CurrentState)
        plot_cutsky_covariance(window_function, CurrentState)

        if CurrentState.settings.show:
            plt.show()

    return window_function


def save_window_function(CurrentState, window_function):
    file = CurrentState.window_file
    with open(file, "wb") as pickle_file:
        pickle.dump(window_function, pickle_file)
    return


def main(CurrentState):
    window = get_window_function(CurrentState)
    save_window_function(CurrentState, window)
    return


if __name__ == "__main__":
    main()

# ----------------------------------------------------- #
