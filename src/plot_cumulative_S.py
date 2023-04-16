import matplotlib.pyplot as plt
from functions import read, plot_cumulative_S
from classes import State


def plotting(CurrentState: State):
    lmax = CurrentState.settings.lmax
    nbins = CurrentState.settings.nbins
    savefig = CurrentState.settings.savefig
    show = CurrentState.settings.show

    files = CurrentState.settings.get_cumu_S_files()
    fig_dir = CurrentState.settings.fig_dir

    plot_cumulative_S(
        read(files[0]),
        read(files[2]),
        read(files[1]),
        read(files[3]),
        nbins,
        lmax,
        "TT",
        "TT",
        fig_dir,
        unit=r"$[\mu K^4]$",
        savefig=savefig,
    )

    plot_cumulative_S(
        read(files[4]),
        read(files[5]),
        read(files[4]),
        read(files[6]),
        nbins,
        lmax,
        "GWGW",
        "GWGW",
        fig_dir,
        unit=r"$[\mu K^4]$",
        savefig=savefig,
    )

    plot_cumulative_S(
        read(files[7]),
        read(files[8]),
        read(files[9]),
        read(files[10]),
        nbins,
        lmax,
        "TGW",
        "TGW",
        fig_dir,
        unit=r"$[\mu K^4]$",
        savefig=savefig,
    )

    plot_cumulative_S(
        read(files[11]),
        read(files[12]),
        read(files[13]),
        read(files[14]),
        nbins,
        lmax,
        "TT_GWGW",
        "TT,GWGW",
        fig_dir,
        savefig=savefig,
    )

    plot_cumulative_S(
        read(files[19]),
        read(files[20]),
        read(files[21]),
        read(files[22]),
        nbins,
        lmax,
        "TT_TGW",
        "TT,TGW",
        fig_dir,
        concl=False,
        savefig=savefig,
    )

    plot_cumulative_S(
        read(files[15]),
        read(files[16]),
        read(files[17]),
        read(files[18]),
        nbins,
        lmax,
        "TT_TGW_GWGW",
        "TT,TGW,GWGW",
        fig_dir,
        concl=True,
        savefig=savefig,
    )

    if show:
        plt.show()


def main(CurrentState: State):
    plotting(CurrentState)
    return


if __name__ == "__main__":
    main()
