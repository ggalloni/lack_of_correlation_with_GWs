import classes as cfg
from src import (
    compute_S_estimators,
    compute_legendre_integrals,
    compute_legendre_polynomials,
    compute_window_function,
    plot_S_histograms,
    plot_angular_correlation_function,
    plot_cumulative_S_estimator,
    plot_cumulative_significance,
    plot_optimal_angular_ranges,
    save_cumulative_S_estimators,
    simulations_production,
    save_optimal,
    save_significance,
    spectra_production,
)


def get_CurrentSettings(**kwargs):
    lmax = 4
    return cfg.Settings(lmax=lmax, **kwargs)


def main(
    *,
    do_simulations: bool = True,
    savefig: bool = False,
    show: bool = False,
    debug: bool = False,
    batch: str = "testing",
    N: int = 100,
):
    print("\n*************** INITIALIZING SETTING *************")
    CurrentSettings = get_CurrentSettings(
        savefig=savefig, show=show, debug=debug, batch=batch, N=N
    )

    if not CurrentSettings.CLS_file.exists():
        print("\n**************** PRODUCING SPECTRA ***************")
        spectra_production.main(CurrentSettings)
    if not CurrentSettings.I_file.exists():
        print("\n********** PRODUCING LEGENDRE INTEGRALS **********")
        compute_legendre_integrals.main(CurrentSettings)

    CurrentState = cfg.State(CurrentSettings)

    if not CurrentState.settings.P_file.exists():
        print("\n********* PRODUCING LEGENDRE POLYNOMIALS *********")
        compute_legendre_polynomials.main(CurrentState)
    if not CurrentState.window_file.exists():
        print("\n*********** PRODUCING WINDOW FUNCTIONS ***********")
        compute_window_function.main(CurrentState)

    if do_simulations:
        print("\n************** PRODUCING SIMULATIONS *************")
        simulations_production.main(CurrentState)
        print("\n************* COMPUTING S ESTIMATORS *************")
        compute_S_estimators.main(CurrentState)
        print("\n******** COMPUTING CUMULATIVE S ESTIMATORS *******")
        save_cumulative_S_estimators.main(CurrentState)
        print("\n********** SAVING OPTIMAL ANGULAR RANGES *********")
        save_optimal.main(CurrentState)
        print("\n************** SAVING SIGNIFICANCES **************")
        save_significance.main(CurrentState)

    if CurrentSettings.show or CurrentSettings.savefig:
        print("\n***** PLOTTING 2-POINT CORRELATION FUNCTIONS *****")
        plot_angular_correlation_function.main(CurrentState)
        print("\n******** PLOTTING S ESTIMATORS HISTOGRAMS ********")
        plot_S_histograms.main(CurrentState)
        print("\n********* PLOTTING OPTIMAL ANGULAR RANGES ********")
        plot_optimal_angular_ranges.main(CurrentState)
        print("\n******** PLOTTING CUMULATIVE S ESTIMATORS ********")
        plot_cumulative_S_estimator.main(CurrentState)
        print("\n************* PLOTTING SIGNIFICANCES *************")
        plot_cumulative_significance.main(CurrentState)
    return


if __name__ == "__main__":
    main()
