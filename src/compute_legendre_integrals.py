"""
The script computes the I(μ) integral for various values of μ.
The values of μ are computed in the script.
"""

import pickle

import numpy as np

from classes import Settings
from functions import legendre_intergal


def compute_I(CurrentSettings: Settings) -> np.ndarray:
    """
    Compute the legengre integrals I.
    """
    return np.array(
        legendre_intergal(
            CurrentSettings.μ_min, CurrentSettings.μ_max, CurrentSettings.lmax
        )
    )


def save_I(CurrentSettings: Settings, I_μs: np.ndarray) -> None:
    """Save the current I_μs to the file CurrentSettings.I_file

    Parameters
    ----------
    CurrentSettings : CurrentSettings
        The settings for the current run.
    I_μs : array_like
        The current in μA.
    """
    with open(CurrentSettings.I_file, "wb") as pickle_file:
        pickle.dump(I_μs, pickle_file)
    return


def main(CurrentSettings: Settings) -> None:
    # compute the value of the current
    I_μs: float = compute_I(CurrentSettings)

    # save the value of the current
    save_I(CurrentSettings, I_μs)

    return


if __name__ == "__main__":
    main()

# ----------------------------------------------------- #
