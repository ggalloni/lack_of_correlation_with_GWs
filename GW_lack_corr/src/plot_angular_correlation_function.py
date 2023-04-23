import pickle

import matplotlib.pyplot as plt

from GW_lack_corr.classes.state import State
from GW_lack_corr.src.common_functions import compute_ang_corr, compute_ang_corr_TT


def plotting(CurrentState: State):
    lmax = CurrentState.settings.lmax
    savefig = CurrentState.settings.savefig
    show = CurrentState.settings.show
    fig_dir = CurrentState.settings.fig_dir

    CLs = CurrentState.CLs
    data = CurrentState.collect_spectra(CurrentState.settings.get_spectra_files())

    file = CurrentState.settings.dir.joinpath(f"lmax{lmax}_P.pkl")
    with open(file, "rb") as pickle_file:
        P = pickle.load(pickle_file)

    compute_ang_corr_TT(
        data["TT_uncon_CL"],
        CurrentState.SMICA_Cl,
        CurrentState.masked_SMICA_Cl,
        CLs,
        P,
        lmax,
        fig_dir,
        step=1,
        savefig=savefig,
    )

    compute_ang_corr(
        data["GWGW_uncon_CL"],
        data["GWGW_con_CL"],
        CLs,
        "GWGW",
        "GWGW",
        P,
        lmax,
        fig_dir,
        step=1,
        savefig=savefig,
    )

    compute_ang_corr(
        data["GWGW_uncon_CL"],
        data["masked_GWGW_con_CL"],
        CLs,
        "masked_GWGW",
        "GWGW",
        P,
        lmax,
        fig_dir,
        step=1,
        savefig=savefig,
    )

    compute_ang_corr(
        data["TGW_uncon_CL"],
        data["TGW_con_CL"],
        CLs,
        "TGW",
        "TGW",
        P,
        lmax,
        fig_dir,
        step=1,
        savefig=savefig,
    )

    compute_ang_corr(
        data["TGW_uncon_CL"],
        data["masked_TGW_con_CL"],
        CLs,
        "masked_TGW",
        "TGW",
        P,
        lmax,
        fig_dir,
        step=1,
        savefig=savefig,
    )

    if show:
        plt.show()


def main(CurrentState: State):
    plotting(CurrentState)
    return


if __name__ == "__main__":
    main()
