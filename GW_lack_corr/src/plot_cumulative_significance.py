import matplotlib.pyplot as plt
from GW_lack_corr.src.common_functions import plot_signi
from GW_lack_corr.classes.state import State

plt.rcParams.update({"figure.max_open_warning": 0})


def plotting(CurrentState: State):
    lmax = CurrentState.settings.lmax
    nbins = CurrentState.settings.nbins
    savefig = CurrentState.settings.savefig
    show = CurrentState.settings.show

    files = CurrentState.settings.get_significance_files()
    data = CurrentState.collect_spectra(files)
    fig_dir = CurrentState.settings.fig_dir

    print("\nPlotting for TT...")
    plot_signi(
        data["signi_TT"],
        data["signi_SMICA"],
        data["signi_masked_SMICA"],
        nbins,
        lmax,
        "TT",
        "TT",
        fig_dir,
        smica=data["signi_SMICA"],
        masked_smica=data["signi_masked_SMICA"],
        savefig=savefig,
    )

    print("\nPlotting for GWGW...")
    plot_signi(
        data["signi_uncon_GWGW"],
        data["signi_con_GWGW"],
        data["signi_masked_con_GWGW"],
        nbins,
        lmax,
        "GWGW",
        "GWGW",
        fig_dir,
        smica=data["signi_SMICA"],
        masked_smica=data["signi_masked_SMICA"],
        savefig=savefig,
    )

    print("\nPlotting for TGW...")
    plot_signi(
        data["signi_uncon_TGW"],
        data["signi_con_TGW"],
        data["signi_masked_con_TGW"],
        nbins,
        lmax,
        "TGW",
        "TGW",
        fig_dir,
        smica=data["signi_SMICA"],
        masked_smica=data["signi_masked_SMICA"],
        savefig=savefig,
    )

    print("\nPlotting for TT, GWGW...")
    plot_signi(
        data["signi_uncon_TT_GWGW"],
        data["signi_con_TT_GWGW"],
        data["signi_masked_con_TT_GWGW"],
        nbins,
        lmax,
        "TT_GWGW",
        "TT,GWGW",
        fig_dir,
        smica=data["signi_SMICA"],
        masked_smica=data["signi_masked_SMICA"],
        savefig=savefig,
    )

    print("\nPlotting for TT, TGW...")
    plot_signi(
        data["signi_uncon_TT_TGW"],
        data["signi_con_TT_TGW"],
        data["signi_masked_con_TT_TGW"],
        nbins,
        lmax,
        "TT_TGW",
        "TT,TGW",
        fig_dir,
        smica=data["signi_SMICA"],
        masked_smica=data["signi_masked_SMICA"],
        savefig=savefig,
    )

    print("\nPlotting for TT, TGW, GWGW...")
    plot_signi(
        data["signi_uncon_TT_TGW_GWGW"],
        data["signi_con_TT_TGW_GWGW"],
        data["signi_masked_con_TT_TGW_GWGW"],
        nbins,
        lmax,
        "TT_TGW_GWGW",
        "TT,TGW,GWGW",
        fig_dir,
        smica=data["signi_SMICA"],
        masked_smica=data["signi_masked_SMICA"],
        savefig=savefig,
    )

    if show:
        plt.show()


def main(CurrentState: State):
    plotting(CurrentState)
    plt.close("all")
    return


if __name__ == "__main__":
    main()
