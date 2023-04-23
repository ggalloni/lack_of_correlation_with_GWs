import numpy as np
import pickle
import pandas as pd

from GW_lack_corr.classes.settings import Settings


def main(CurrentSettings: Settings):
    lmaxes = CurrentSettings.lmaxes
    tables_file = CurrentSettings.dir.joinpath("tables.txt")
    angles = {}
    for i, max in enumerate(lmaxes):
        file = CurrentSettings.dir.joinpath(f"lmax{max}_opt_angs.pkl")
        with open(file, "rb") as pickle_file:
            angles[max] = pickle.load(pickle_file)

    signis = {}
    for i, max in enumerate(lmaxes):
        file = CurrentSettings.dir.joinpath(f"lmax{max}_signi_dict.pkl")
        with open(file, "rb") as pickle_file:
            signis[max] = pickle.load(pickle_file)

    # ----------------------------------------------------- #

    def mergeDictionary(dict_1, dict_2, dict_3):
        dict_4 = {}
        for key in dict_1.keys():
            dict_4[key + "min"] = [
                f"${dict_1[key][0]}^\\circ$",
                f"${dict_2[key][0]}^\\circ$",
                f"${dict_3[key][0]}^\\circ$",
            ]
            dict_4[key + "max"] = [
                f"${dict_1[key][1]}^\\circ$",
                f"${dict_2[key][1]}^\\circ$",
                f"${dict_3[key][1]}^\\circ$",
            ]
            dict_4[key + "PD"] = [
                f"${dict_1[key][2]}\\%$",
                f"${dict_2[key][2]}\\%$",
                f"${dict_3[key][2]}\\%$",
            ]
        return dict_4

    opt_angs = mergeDictionary(angles[4], angles[6], angles[10])
    opt_angs["lmax"] = [
        "$\\ell_{\rm max} = 4$",
        "$\\ell_{\rm max} = 6$",
        "$\\ell_{\rm max} = 10$",
    ]

    dataframe = pd.DataFrame(opt_angs)
    n_rows, n_cols = dataframe.shape
    dataframe = dataframe.set_index([[""] * n_rows])

    columns = np.roll(list(opt_angs.keys()), 1)

    columns = [
        "lmax",
        "GWGWmin",
        "GWGWmax",
        "TGWmin",
        "TGWmax",
        "TT_GWGWmin",
        "TT_GWGWmax",
        "TT_TGWmin",
        "TT_TGWmax",
        "TT_TGW_GWGWmin",
        "TT_TGW_GWGWmax",
    ]

    dataframe_tex = dataframe.to_latex(
        index=True,
        escape=False,
        column_format="r" * n_cols,
        header=False,
        columns=columns,
    )
    dataframe_tex = dataframe_tex.replace("{} &", "")
    with open(tables_file, "w") as f:
        f.write("******** FULLSKY OPTIMAL ANGLES ********\n\n")
        f.write(dataframe_tex)

    columns = [
        "lmax",
        "masked_GWGWmin",
        "masked_GWGWmax",
        "masked_TGWmin",
        "masked_TGWmax",
        "masked_TT_GWGWmin",
        "masked_TT_GWGWmax",
        "masked_TT_TGWmin",
        "masked_TT_TGWmax",
        "masked_TT_TGW_GWGWmin",
        "masked_TT_TGW_GWGWmax",
    ]

    dataframe_tex = dataframe.to_latex(
        index=True,
        escape=False,
        column_format="r" * n_cols,
        header=False,
        columns=columns,
    )
    dataframe_tex = dataframe_tex.replace("{} &", "")
    with open(tables_file, "a") as f:
        f.write("\n******** MASKED OPTIMAL ANGLES ********\n\n")
        f.write(dataframe_tex)

    columns = [
        "lmax",
        "GWGWPD",
        "TGWPD",
        "TT_GWGWPD",
        "TT_TGWPD",
        "TT_TGW_GWGWPD",
    ]

    dataframe_tex = dataframe.to_latex(
        index=True,
        escape=False,
        column_format="r" * n_cols,
        header=False,
        columns=columns,
    )
    dataframe_tex = dataframe_tex.replace("{} &", "")
    with open(tables_file, "a") as f:
        f.write("\n******** FULLSKY PERCENTAGE DISPLECEMENTS ********\n\n")
        f.write(dataframe_tex)

    columns = [
        "lmax",
        "masked_GWGWPD",
        "masked_TGWPD",
        "masked_TT_GWGWPD",
        "masked_TT_TGWPD",
        "masked_TT_TGW_GWGWPD",
    ]

    dataframe_tex = dataframe.to_latex(
        index=True,
        escape=False,
        column_format="r" * n_cols,
        header=False,
        columns=columns,
    )
    dataframe_tex = dataframe_tex.replace("{} &", "")
    with open(tables_file, "a") as f:
        f.write("\n******** MASKED PERCENTAGE DISPLECEMENTS ********\n\n")
        f.write(dataframe_tex)

    def mergeDictionary(dict_1, dict_2, dict_3):
        dict_4 = {}
        for key in dict_1.keys():
            if key == "TT":
                dict_4[key] = [
                    f"${dict_1[key][0]}\\sigma$",
                    f"${dict_2[key][0]}\\sigma$",
                    f"${dict_3[key][0]}\\sigma$",
                ]
                dict_4["masked" + key] = [
                    f"${dict_1[key][1]}\\sigma$",
                    f"${dict_2[key][1]}\\sigma$",
                    f"${dict_3[key][1]}\\sigma$",
                ]
            else:
                dict_4[key] = [
                    f"${dict_1[key][0]}\\%$",
                    f"${dict_2[key][0]}\\%$",
                    f"${dict_3[key][0]}\\%$",
                ]
                dict_4["masked" + key] = [
                    f"${dict_1[key][1]}\\%$",
                    f"${dict_2[key][1]}\\%$",
                    f"${dict_3[key][1]}\\%$",
                ]
        return dict_4

    significances = mergeDictionary(signis[4], signis[6], signis[10])

    significances["lmax"] = [
        "$\\ell_{\rm max} = 4$",
        "$\\ell_{\rm max} = 6$",
        "$\\ell_{\rm max} = 10$",
    ]

    dataframe = pd.DataFrame(significances)
    n_rows, n_cols = dataframe.shape
    dataframe = dataframe.set_index([[""] * n_rows])

    columns = np.roll(list(significances.keys()), 1)

    columns = [
        "lmax",
        "TT",
        "GWGW",
        "TGW",
        "TT_GWGW",
        "TT_TGW",
        "TT_TGW_GWGW",
    ]

    dataframe_tex = dataframe.to_latex(
        index=True,
        escape=False,
        column_format="r" * n_cols,
        header=False,
        columns=columns,
    )
    dataframe_tex = dataframe_tex.replace("{} &", "")
    with open(tables_file, "a") as f:
        f.write("\n******** FULLSKY SIGNIFICANCES ********\n\n")
        f.write(dataframe_tex)

    columns = [
        "lmax",
        "maskedTT",
        "maskedGWGW",
        "maskedTGW",
        "maskedTT_GWGW",
        "maskedTT_TGW",
        "maskedTT_TGW_GWGW",
    ]

    dataframe_tex = dataframe.to_latex(
        index=True,
        escape=False,
        column_format="r" * n_cols,
        header=False,
        columns=columns,
    )
    dataframe_tex = dataframe_tex.replace("{} &", "")
    with open(tables_file, "a") as f:
        f.write("\n******** MASKED SIGNIFICANCES ********\n\n")
        f.write(dataframe_tex)
    return


if __name__ == "__main__":
    main()
