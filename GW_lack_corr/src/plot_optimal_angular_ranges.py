import matplotlib.pyplot as plt

from GW_lack_corr.src.common_functions import optimal_ang
from GW_lack_corr.classes.state import State


def plotting(CurrentState: State):
    lmax = CurrentState.settings.lmax
    savefig = CurrentState.settings.savefig
    show = CurrentState.settings.show
    files = CurrentState.settings.get_S_files()
    cmap = CurrentState.settings.cmap
    fig_dir = CurrentState.settings.fig_dir

    print("\nPlotting optimal angles for TT...")
    optimal_ang(
        files[0], files[2], lmax, "TT", "TT", cmap, fig_dir, savefig=savefig, show=False
    )

    print("\nPlotting optimal angles for masked TT...")
    optimal_ang(
        files[1],
        files[3],
        lmax,
        "masked_TT",
        "TT",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for TGW...")
    optimal_ang(
        files[11],
        files[12],
        lmax,
        "TGW",
        "TGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for masked TGW...")
    optimal_ang(
        files[13],
        files[14],
        lmax,
        "masked_TGW",
        "TGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for GWGW...")
    optimal_ang(
        files[4],
        files[5],
        lmax,
        "GWGW",
        "GWGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for masked GWGW...")
    optimal_ang(
        files[4],
        files[6],
        lmax,
        "masked_GWGW",
        "GWGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for TT, GWGW...")
    optimal_ang(
        files[7],
        files[8],
        lmax,
        "TT_GWGW",
        "TT, GWGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for masked TT, GWGW...")
    optimal_ang(
        files[9],
        files[10],
        lmax,
        "masked_TT_GWGW",
        "TT, GWGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for TT, TGW...")
    optimal_ang(
        files[19],
        files[20],
        lmax,
        "TT_TGW",
        "TT, TGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for masked TT, TGW...")
    optimal_ang(
        files[21],
        files[22],
        lmax,
        "masked_TT_TGW",
        "TT, TGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for TT, TGW, GWGW...")
    optimal_ang(
        files[15],
        files[16],
        lmax,
        "TT_TGW_GWGW",
        "TT, TGW, GWGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    print("\nPlotting optimal angles for masked TT, TGW, GWGW...")
    optimal_ang(
        files[17],
        files[18],
        lmax,
        "masked_TT_TGW_GWGW",
        "TT, TGW, GWGW",
        cmap,
        fig_dir,
        savefig=savefig,
        show=False,
    )

    if show:
        plt.show()
    return


def main(CurrentState: State):
    plotting(CurrentState)
    return


if __name__ == "__main__":
    main()
