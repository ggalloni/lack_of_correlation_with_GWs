import matplotlib.pyplot as plt
from GW_lack_corr.src.common_functions import plot_hist, plot_joint_hist
from GW_lack_corr.classes.state import State
import pickle

plt.rcParams.update({"figure.max_open_warning": 0})


def plot_histograms(CurrentState: State):
    with open(CurrentState.settings.opt_angs_file, "rb") as pickle_file:
        opt_angs = pickle.load(pickle_file)
    lmax = CurrentState.settings.lmax
    files = CurrentState.settings.get_S_files()
    savefig = CurrentState.settings.savefig
    show = CurrentState.settings.show
    fig_dir = CurrentState.settings.fig_dir

    print("\nPlotting S for TT...")
    plot_hist(opt_angs, "TT", "TT", lmax, files[0], files[2], fig_dir, savefig=savefig)

    print("\nPlotting S for masked TT...")
    plot_hist(
        opt_angs, "masked_TT", "TT", lmax, files[1], files[3], fig_dir, savefig=savefig
    )

    print("\nPlotting S for GWGW...")
    plot_hist(
        opt_angs, "GWGW", "GWGW", lmax, files[4], files[5], fig_dir, savefig=savefig
    )

    print("\nPlotting S for masked GWGW...")
    plot_hist(
        opt_angs,
        "masked_GWGW",
        "GWGW",
        lmax,
        files[4],
        files[6],
        fig_dir,
        savefig=savefig,
    )

    print("\nPlotting S for TGW...")
    plot_hist(
        opt_angs, "TGW", "TGW", lmax, files[11], files[12], fig_dir, savefig=savefig
    )

    print("\nPlotting S for masked TGW...")
    plot_hist(
        opt_angs,
        "masked_TGW",
        "TGW",
        lmax,
        files[13],
        files[14],
        fig_dir,
        savefig=savefig,
    )

    print("\nPlotting S for the TT, GWGW...")
    plot_joint_hist(
        opt_angs, "TGW", "TT,GWGW", lmax, files[7], files[8], fig_dir, savefig=savefig
    )

    print("\nPlotting S for the masked TT, GWGW...")
    plot_joint_hist(
        opt_angs,
        "masked_TGW",
        "TT,GWGW",
        lmax,
        files[9],
        files[10],
        fig_dir,
        savefig=savefig,
    )

    print("\nPlotting S for the TT, TGW...")
    plot_joint_hist(
        opt_angs,
        "TT_TGW",
        "TT,TGW",
        lmax,
        files[19],
        files[20],
        fig_dir,
        savefig=savefig,
    )

    print("\nPlotting S for the masked TT, TGW...")
    plot_joint_hist(
        opt_angs,
        "masked_TT_TGW",
        "TT,TGW",
        lmax,
        files[21],
        files[22],
        fig_dir,
        concl=True,
        savefig=savefig,
    )

    print("\nPlotting S for the TT, TGW, GWGW...")
    plot_joint_hist(
        opt_angs,
        "TT_TGW_GWGW",
        "TT,TGW,GWGW",
        lmax,
        files[15],
        files[16],
        fig_dir,
        savefig=savefig,
    )

    print("\nPlotting S for the masked TT, TGW, GWGW...")
    plot_joint_hist(
        opt_angs,
        "masked_TT_TGW_GWGW",
        "TT,TGW,GWGW",
        lmax,
        files[17],
        files[18],
        fig_dir,
        savefig=savefig,
    )

    if show:
        plt.show()


def main(CurrentState: State):
    plot_histograms(CurrentState)
    plt.close("all")
    return


if __name__ == "__main__":
    main()
