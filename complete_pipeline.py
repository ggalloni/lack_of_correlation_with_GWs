import run_lmax4
import run_lmax6
import run_lmax10
from src import make_tables


def main():
    run_everything = True
    savefig = True
    show = False
    debug = False
    batch = "testing"
    N = 100

    if run_everything:
        do_simulations = True

        print("\n***************** RUNNING LMAX = 4 PIPELINE *****************")
        run_lmax4.main(
            do_simulations=do_simulations,
            savefig=savefig,
            show=show,
            debug=debug,
            batch=batch,
            N=N,
        )
        print("\n***************** RUNNING LMAX = 6 PIPELINE *****************")
        run_lmax6.main(
            do_simulations=do_simulations,
            savefig=savefig,
            show=show,
            debug=debug,
            batch=batch,
            N=N,
        )
        print("\n***************** RUNNING LMAX = 10 PIPELINE ****************")
        run_lmax10.main(
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
