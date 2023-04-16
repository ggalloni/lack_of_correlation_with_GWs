"""
The script computes the I(μ) integral for various values of μ.
The values of μ are computed in the script.
"""

import numpy as np
import pickle
from functions import legendre_intergal


def compute_I(CurrentSettings):
    return np.array(
        legendre_intergal(
            CurrentSettings.μ_min, CurrentSettings.μ_max, CurrentSettings.lmax
        )
    )


def save_I(CurrentSettings, I_μs):
    with open(CurrentSettings.I_file, "wb") as pickle_file:
        pickle.dump(I_μs, pickle_file)
    return


def main(CurrentSettings):
    I_μs = compute_I(CurrentSettings)
    save_I(CurrentSettings, I_μs)

    return


if __name__ == "__main__":
    main()

# ----------------------------------------------------- #
