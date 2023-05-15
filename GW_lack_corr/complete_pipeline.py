from GW_lack_corr.individual_runs import run_lmax4, run_lmax6, run_lmax10
from GW_lack_corr.src import make_tables


def main():
    """
    This is the main function for the complete pipeline. It will run the pipeline
    for lmax = 4, lmax = 6, and lmax = 10, and then make the tables.

    Parameters
    ----------
    produce_new_seeds : bool
        Set to True to produce new seeds. Default is False. This will overwrite existing seeds and affect only the stochastic production of simulations. The rest of the analysis is deterministic. In the current implementation, seeds are shared by the three lmax runs.

    run_everything : bool
        Set to False to only make tables. Default is True.

    savefig : bool
        Set to False to not save figures. Default is True.

    show : bool
        Set to True to show figures. Default is False.

    debug : bool
        Set to True to run in debug mode (note that this will avoid overwriting existing data).
        Default is False.

    batch : str
        This is the name of the folder where the data will be saved. Default is 'testing'.

    N : int
        Number of simulations to run. Default is 100.

    do_simulations : bool
        Set to False to not run simulations. Default is True. Note that this will also recompute the S estimators, sum them over angular configuration, etc etc.
    """
    produce_new_seeds = False
    run_everything = True
    savefig = True
    show = False  # Careful in setting this to True, as it will open a lot of figures.
    debug = False
    batch = "testing"
    N = 100

    if run_everything:
        do_simulations = True

        print("\n***************** RUNNING LMAX = 4 PIPELINE *****************")
        run_lmax4.main(
            produce_new_seeds=produce_new_seeds,
            do_simulations=do_simulations,
            savefig=savefig,
            show=show,
            debug=debug,
            batch=batch,
            N=N,
        )
        print("\n***************** RUNNING LMAX = 6 PIPELINE *****************")
        run_lmax6.main(
            produce_new_seeds=produce_new_seeds,
            do_simulations=do_simulations,
            savefig=savefig,
            show=show,
            debug=debug,
            batch=batch,
            N=N,
        )
        print("\n***************** RUNNING LMAX = 10 PIPELINE ****************")
        run_lmax10.main(
            produce_new_seeds=produce_new_seeds,
            do_simulations=do_simulations,
            savefig=savefig,
            show=show,
            debug=debug,
            batch=batch,
            N=N,
        )

    print("\n******************* MAKING RESULTS TABLES *******************")
    CurrentSettings = run_lmax4.get_CurrentSettings(
        savefig=savefig,
        show=show,
        debug=debug,
        batch=batch,
        N=N,
    )
    make_tables.main(CurrentSettings)
    print("\n********* SAVED TABLES TO tables.txt IN DATA FOLDER *********")
    return


if __name__ == "__main__":
    main()
    print("\n°`°º¤ø,,ø¤°º¤ø,,ø¤º°`° EVERYTHING DONE! °`°º¤ø,,ø¤°º¤ø,,ø¤º°`°")
