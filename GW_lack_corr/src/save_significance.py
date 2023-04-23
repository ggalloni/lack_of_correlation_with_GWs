import pickle

from GW_lack_corr.src.common_functions import save_sigmas, save_signi
from GW_lack_corr.classes.state import State


def saving_significance_arrays(CurrentState: State):
    in_files = CurrentState.settings.get_cumu_S_files()
    out_files = CurrentState.settings.get_significance_files()

    print("\nSaving significance arrays...")

    print("\nSMICA...")
    save_sigmas(out_files[2], in_files[0], in_files[2])
    print("\nMasked SMICA...")
    save_sigmas(out_files[3], in_files[0], in_files[3])
    print("\nTT...")
    save_sigmas(out_files[0], in_files[0], in_files[0])
    print("\nMasked TT...")
    save_sigmas(out_files[1], in_files[1], in_files[1])
    print("\nConstrained GWGW...")
    save_sigmas(out_files[5], in_files[4], in_files[5])
    print("\nMasked constrained GWGW...")
    save_sigmas(out_files[6], in_files[4], in_files[6])
    print("\nUnconstrained GWGW...")
    save_sigmas(out_files[4], in_files[4], in_files[4])
    print("\nConstrained TT, GWGW...")
    save_sigmas(out_files[8], in_files[7], in_files[8])
    print("\nUnconstrained TT, GWGW...")
    save_sigmas(out_files[7], in_files[7], in_files[7])
    print("\nMasked constrained TT, GWGW...")
    save_sigmas(out_files[10], in_files[9], in_files[10])
    print("\nMasked unconstrained TT, GWGW...")
    save_sigmas(out_files[9], in_files[9], in_files[9])
    print("\nConstrained TGW...")
    save_sigmas(out_files[12], in_files[11], in_files[12])
    print("\nUnconstrained TGW...")
    save_sigmas(out_files[11], in_files[11], in_files[11])
    print("\nMasked constrained TGW...")
    save_sigmas(out_files[14], in_files[13], in_files[14])
    print("\nMasked unconstrained TGW...")
    save_sigmas(out_files[13], in_files[13], in_files[13])
    print("\nConstrained TT, TGW...")
    save_sigmas(out_files[20], in_files[19], in_files[20])
    print("\nUnconstrained TT, TGW...")
    save_sigmas(out_files[19], in_files[19], in_files[19])
    print("\nMasked constrained TT, TGW...")
    save_sigmas(out_files[22], in_files[21], in_files[22])
    print("\nMasked unconstrained TT, TGW...")
    save_sigmas(out_files[21], in_files[21], in_files[21])
    print("\nConstrained TT, TGW, GWGW...")
    save_sigmas(out_files[16], in_files[15], in_files[16])
    print("\nUnconstrained TT, TGW, GWGW...")
    save_sigmas(out_files[15], in_files[15], in_files[15])
    print("\nMasked constrained TT, TGW, GWGW...")
    save_sigmas(out_files[18], in_files[17], in_files[18])
    print("\nMasked unconstrained TT, TGW, GWGW...")
    save_sigmas(out_files[17], in_files[17], in_files[17])


def save_significance_greater_that_SMICA(CurrentState: State):
    files = CurrentState.settings.get_significance_files()
    data = CurrentState.collect_spectra(files)
    signi_dict = CurrentState.signi_dict

    save_signi(
        data["signi_TT"],
        data["signi_SMICA"],
        data["signi_masked_SMICA"],
        "TT",
        smica=data["signi_SMICA"],
        signi_dict=signi_dict,
        masked_smica=data["signi_masked_SMICA"],
    )

    save_signi(
        data["signi_uncon_GWGW"],
        data["signi_con_GWGW"],
        data["signi_masked_con_GWGW"],
        "GWGW",
        smica=data["signi_SMICA"],
        signi_dict=signi_dict,
        masked_smica=data["signi_masked_SMICA"],
    )

    save_signi(
        data["signi_uncon_TGW"],
        data["signi_con_TGW"],
        data["signi_masked_con_TGW"],
        "TGW",
        smica=data["signi_SMICA"],
        signi_dict=signi_dict,
        masked_smica=data["signi_masked_SMICA"],
    )

    save_signi(
        data["signi_uncon_TT_GWGW"],
        data["signi_con_TT_GWGW"],
        data["signi_masked_con_TT_GWGW"],
        "TT_GWGW",
        smica=data["signi_SMICA"],
        signi_dict=signi_dict,
        masked_smica=data["signi_masked_SMICA"],
    )

    save_signi(
        data["signi_uncon_TT_TGW_GWGW"],
        data["signi_con_TT_TGW_GWGW"],
        data["signi_masked_con_TT_TGW_GWGW"],
        "TT_TGW_GWGW",
        smica=data["signi_SMICA"],
        signi_dict=signi_dict,
        masked_smica=data["signi_masked_SMICA"],
    )

    save_signi(
        data["signi_uncon_TT_TGW"],
        data["signi_con_TT_TGW"],
        data["signi_masked_con_TT_TGW"],
        "TT_TGW",
        smica=data["signi_SMICA"],
        signi_dict=signi_dict,
        masked_smica=data["signi_masked_SMICA"],
    )

    # ----------------------------------------------------------------#

    file = CurrentState.settings.dir.joinpath(
        f"lmax{CurrentState.settings.lmax}_signi_dict.pkl"
    )
    with open(file, "wb") as pickle_file:
        pickle.dump(signi_dict, pickle_file)
    # ----------------------------------------------------- #
    return


def main(CurrentState: State):
    saving_significance_arrays(CurrentState)
    save_significance_greater_that_SMICA(CurrentState)
    return


if __name__ == "__main__":
    main()
