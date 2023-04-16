import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss
from classes import State
from tqdm import tqdm


def compute_legendre_polynomials():
    lmax = 2000
    start = 180 / (lmax - 1)
    step = 0.1
    angs = np.arange(start, 180 + step, step)
    ell = np.arange(0, lmax + 1, 1)
    P = []
    for i in tqdm(range(len(angs))):
        P.append(ss.eval_legendre(ell, np.cos(np.deg2rad(angs[i]))))
    return P


def main(CurrentState: State):
    lmax = 2000
    start = 180 / (lmax - 1)
    step = 0.1
    angs = np.arange(start, 180 + step, step)
    ell = np.arange(0, lmax + 1, 1)

    savefig = CurrentState.settings.savefig
    show = CurrentState.settings.show
    fig_dir = CurrentState.settings.fig_dir

    CLs = CurrentState.CLs
    P = compute_legendre_polynomials()

    ang_tt = (
        np.array((2 * ell + 1) * CLs["tt"][: lmax + 1]) @ np.array(P).T / (4 * np.pi)
    )

    ang_gwgw = (
        np.array((2 * ell + 1) * CLs["gwgw"][: lmax + 1]) @ np.array(P).T / (4 * np.pi)
    )

    ang_tgw = (
        np.array((2 * ell + 1) * CLs["tgw"][: lmax + 1]) @ np.array(P).T / (4 * np.pi)
    )

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.plot(
        angs,
        ang_tt,
        color="red",
        lw=2,
        label=r"TT",
        zorder=20,
    )

    plt.ylim(None, 12000)
    plt.xlim(0.1, 180)
    plt.axhline(0, color="black", ls="--", lw=2, zorder=-20)

    plt.xlabel(r"$\theta\ [^\circ]$", fontsize=17)
    plt.ylabel(r"$C^{TT}(\theta)\ [\mu K^2]$", fontsize=17)
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(direction="in", which="both", labelsize=14, width=2, zorder=101)
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    plt.semilogx()
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath(f"ang_corr_TT_lmax{lmax}.png"), dpi=150)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.plot(
        angs,
        ang_gwgw,
        color="dodgerblue",
        lw=2,
        label=r"GWGW",
        zorder=20,
    )

    plt.ylim(None, 150000)
    plt.xlim(0.1, 180)
    plt.axhline(0, color="black", ls="--", lw=2, zorder=-20)

    plt.xlabel(r"$\theta\ [^\circ]$", fontsize=17)
    plt.ylabel(r"$C^{GWGW}(\theta)\ [\mu K^2]$", fontsize=17)
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(direction="in", which="both", labelsize=14, width=2, zorder=101)
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    plt.semilogx()
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath(f"ang_corr_GWGW_lmax{lmax}.png"), dpi=150)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))

    plt.plot(
        angs,
        ang_tgw,
        color="forestgreen",
        lw=2,
        label=r"TGW",
        zorder=20,
    )

    plt.ylim(None, 20000)
    plt.xlim(0.1, 180)
    plt.axhline(0, color="black", ls="--", lw=2, zorder=-20)

    plt.xlabel(r"$\theta\ [^\circ]$", fontsize=17)
    plt.ylabel(r"$C^{TGW}(\theta)\ [\mu K^2]$", fontsize=17)
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(direction="in", which="both", labelsize=14, width=2, zorder=101)
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    plt.semilogx()
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_dir.joinpath(f"ang_corr_TGW_lmax{lmax}.png"), dpi=150)
    if show:
        plt.show()


if __name__ == "__main__":
    main()
