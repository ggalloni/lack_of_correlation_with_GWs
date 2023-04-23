import pickle

from GW_lack_corr.src.common_functions import save_optimal
from GW_lack_corr.classes.state import State


def get_optimal_angles(CurrentState: State):
    lmax = CurrentState.settings.lmax
    files = CurrentState.settings.get_S_files()

    vects = {}
    vects["uncon"] = [
        files[0],
        files[1],
        files[4],
        files[4],
        files[7],
        files[9],
        files[11],
        files[13],
        files[15],
        files[17],
        files[19],
        files[21],
    ]
    vects["con"] = [
        files[2],
        files[3],
        files[5],
        files[6],
        files[8],
        files[10],
        files[12],
        files[14],
        files[16],
        files[18],
        files[20],
        files[22],
    ]
    vects["field"] = [
        "TT",
        "masked_TT",
        "GWGW",
        "masked_GWGW",
        "TT_GWGW",
        "masked_TT_GWGW",
        "TGW",
        "masked_TGW",
        "TT_TGW_GWGW",
        "masked_TT_TGW_GWGW",
        "TT_TGW",
        "masked_TT_TGW",
    ]

    opt_angs = {}
    for uncon, con, field in zip(vects["uncon"], vects["con"], vects["field"]):
        opt_angs[field] = save_optimal(uncon, con, lmax)

    CurrentState.opt_angs = opt_angs
    with open(CurrentState.settings.opt_angs_file, "wb") as pickle_file:
        pickle.dump(opt_angs, pickle_file)
    return


def main(CurrentState: State):
    get_optimal_angles(CurrentState)
    return


if __name__ == "__main__":
    main()
