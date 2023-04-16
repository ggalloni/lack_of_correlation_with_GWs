from classy import Class
import pickle
from classes import Settings


def main(CurrentSettings: Settings):
    lmax = 2000

    params = {
        "YHe": "0.2454006",
        "omega_b": "0.0223828",
        "omega_cdm": "0.1201075",
        "T_ncdm": 0.7137658555036082,
        "H0": "67.32117",
        "tau_reio": "0.05430842",
        "A_s": "2.100549e-09",
        "T_cmb": "2.7255",
        "non linear": "halofit",
        "output": "tCl,pCl,lCl,mPk,gwCl",
        "modes": "s, t",
        "N_ur": "2.046",
        "N_ncdm": "1",
        "m_ncdm": "0.06",
        "gauge": "newtonian",
        "n_s": "0.9660499",
        "r": 1e-10,
        "format": "class",
        "l_cgwb_max": lmax,
    }

    M = Class()
    M.set(params)
    M.compute()
    CLs = M.raw_cl(lmax)

    norm = M.T_cmb() ** 2 * 1e12
    CLs["tt"] *= norm
    CLs["gwgw"] *= norm
    CLs["tgw"] *= norm

    with open(CurrentSettings.CLS_file, "wb") as pickle_file:
        pickle.dump(CLs, pickle_file)
    return


if __name__ == "__main__":
    main()
