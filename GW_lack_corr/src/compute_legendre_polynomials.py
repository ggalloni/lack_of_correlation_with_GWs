import numpy as np
import pickle
import scipy.special as ss
from GW_lack_corr.classes.state import State


def save_legendre_polynomials(CurrentState: State):
    lmax = CurrentState.settings.lmax
    angs = np.arange(
        CurrentState.settings.start,
        180 + CurrentState.settings.step,
        CurrentState.settings.step,
    )
    ell = np.arange(0, lmax + 1, 1)
    P = []
    for i in range(len(angs)):
        P.append(ss.eval_legendre(ell, np.cos(np.deg2rad(angs[i]))))

    file = CurrentState.settings.P_file
    with open(file, "wb") as pickle_file:
        pickle.dump(P, pickle_file)
    return


def main(CurrentState: State):
    save_legendre_polynomials(CurrentState)
    return


if __name__ == "__main__":
    main()
