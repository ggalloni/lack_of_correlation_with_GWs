from pathlib import Path
import classes as cfg
from src import (
    spectra_prod,
    compute_I,
    compute_window,
    sims_production,
    compute_S,
    save_cumulative_S,
    save_optimal,
    compute_Legendre,
    save_significance,
    plot_ang_corr,
    plot_S_hist,
    plot_optimal,
    plot_cumulative_S,
    plot_cumulative_signi,
)


def get_CurrentSettings(**kwargs):
    lmax = 10
    return cfg.Settings(lmax=lmax, **kwargs)


def main(
    do_simulations=True, savefig=True, show=False, debug=False, batch="testing", N=100
):
    print("\n*************** INITIALIZING SETTING *************")
    CurrentSettings = get_CurrentSettings(
        savefig=savefig, show=show, debug=debug, batch=batch, N=N
    )

    if not CurrentSettings.CLS_file.exists():
        print("\n**************** PRODUCING SPECTRA ***************")
        spectra_prod.main(CurrentSettings)
    if not CurrentSettings.I_file.exists():
        print("\n********** PRODUCING LEGENDRE INTEGRALS **********")
        compute_I.main(CurrentSettings)

    CurrentState = cfg.State(CurrentSettings)

    if not CurrentState.settings.P_file.exists():
        print("\n********* PRODUCING LEGENDRE POLYNOMIALS *********")
        compute_Legendre.main(CurrentState)
    if not CurrentState.window_file.exists():
        print("\n*********** PRODUCING WINDOW FUNCTIONS ***********")
        compute_window.main(CurrentState)

    if do_simulations:
        print("\n************** PRODUCING SIMULATIONS *************")
        sims_production.main(CurrentState)
        print("\n************* COMPUTING S ESTIMATORS *************")
        compute_S.main(CurrentState)
        print("\n******** COMPUTING CUMULATIVE S ESTIMATORS *******")
        save_cumulative_S.main(CurrentState)
    print("\n********** SAVING OPTIMAL ANGULAR RANGES *********")
    save_optimal.main(CurrentState)
    print("\n************** SAVING SIGNIFICANCES **************")
    save_significance.main(CurrentState)

    if CurrentSettings.show or CurrentSettings.savefig:
        print("\n***** PLOTTING 2-POINT CORRELATION FUNCTIONS *****")
        plot_ang_corr.main(CurrentState)
        print("\n******** PLOTTING S ESTIMATORS HISTOGRAMS ********")
        plot_S_hist.main(CurrentState)
        print("\n********* PLOTTING OPTIMAL ANGULAR RANGES ********")
        plot_optimal.main(CurrentState)
        print("\n******** PLOTTING CUMULATIVE S ESTIMATORS ********")
        plot_cumulative_S.main(CurrentState)
        print("\n************* PLOTTING SIGNIFICANCES *************")
        plot_cumulative_signi.main(CurrentState)
    return


if __name__ == "__main__":
    main()
