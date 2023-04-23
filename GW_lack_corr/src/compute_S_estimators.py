from GW_lack_corr.classes.state import State


def save_single_estimators(CurrentState: State) -> None:
    """
    Compute and save the estimators for each field.

    Parameters
    ----------
    CurrentState : State
        The current state of the pipeline.
    """
    from GW_lack_corr.src.common_functions import compute_S

    integral = CurrentState.get_legendre_integrals()
    in_files = CurrentState.settings.get_spectra_files()
    out_files = CurrentState.settings.get_S_files()

    spectra = CurrentState.collect_spectra(in_files)

    μ_min = CurrentState.settings.μ_min
    μ_max = CurrentState.settings.μ_max
    lmax = CurrentState.settings.lmax
    N = CurrentState.settings.N

    print("\nComputing S estimators...")

    print("\nTT...")
    compute_S(out_files[0], μ_min, μ_max, integral, spectra["TT_uncon_CL"], N, lmax)

    print("\nMasked TT...")
    compute_S(
        out_files[1], μ_min, μ_max, integral, spectra["masked_TT_uncon_CL"], N, lmax
    )

    print("\nSMICA...")
    compute_S(out_files[2], μ_min, μ_max, integral, CurrentState.SMICA_Cl, 1, lmax)

    print("\nMasked SMICA...")
    compute_S(
        out_files[3], μ_min, μ_max, integral, CurrentState.masked_SMICA_Cl, 1, lmax
    )

    print("\nUnconstrained GWGW...")
    compute_S(out_files[4], μ_min, μ_max, integral, spectra["GWGW_uncon_CL"], N, lmax)

    print("\nConstrained GWGW...")
    compute_S(out_files[5], μ_min, μ_max, integral, spectra["GWGW_con_CL"], N, lmax)

    print("\nMasked constrained GWGW...")
    compute_S(
        out_files[6], μ_min, μ_max, integral, spectra["masked_GWGW_con_CL"], N, lmax
    )

    print("\nUnconstrained TGW...")
    compute_S(out_files[11], μ_min, μ_max, integral, spectra["TGW_uncon_CL"], N, lmax)

    print("\nConstrained TGW...")
    compute_S(out_files[12], μ_min, μ_max, integral, spectra["TGW_con_CL"], N, lmax)

    print("\nMasked unconstrained TGW...")
    compute_S(
        out_files[13], μ_min, μ_max, integral, spectra["masked_TGW_uncon_CL"], N, lmax
    )

    print("\nMasked constrained TGW...")
    compute_S(
        out_files[14], μ_min, μ_max, integral, spectra["masked_TGW_con_CL"], N, lmax
    )
    return


def save_combined_estimators(out_files: list) -> None:
    """Save combined estimators

    Parameters
    ----------
        out_files: list
            List of output files
    """
    from GW_lack_corr.src.common_functions import compute_joint_S, compute_super_S

    print("\nUnconstrained TT, GWGW...")
    compute_joint_S(
        out_files[7], out_files[0], out_files[0], out_files[4], out_files[4]
    )

    print("\nConstrained TT, GWGW...")
    compute_joint_S(
        out_files[8], out_files[0], out_files[2], out_files[4], out_files[5]
    )

    print("\nMasked unconstrained TT, GWGW...")
    compute_joint_S(
        out_files[9], out_files[1], out_files[1], out_files[4], out_files[4]
    )

    print("\nMasked constrained TT, GWGW...")
    compute_joint_S(
        out_files[10], out_files[1], out_files[3], out_files[4], out_files[6]
    )

    print("\nUnconstrained TT, TGW...")
    compute_joint_S(
        out_files[19], out_files[0], out_files[0], out_files[11], out_files[11]
    )

    print("\nConstrained TT, TGW...")
    compute_joint_S(
        out_files[20], out_files[0], out_files[2], out_files[11], out_files[12]
    )

    print("\nMasked unconstrained TT, TGW...")
    compute_joint_S(
        out_files[21], out_files[1], out_files[1], out_files[13], out_files[13]
    )

    print("\nMasked constrained TT, TGW...")
    compute_joint_S(
        out_files[22], out_files[1], out_files[3], out_files[13], out_files[14]
    )

    print("\nUnconstrained TT, TGW, GWGW...")
    compute_super_S(
        out_files[15],
        out_files[0],
        out_files[0],
        out_files[4],
        out_files[4],
        out_files[11],
        out_files[11],
    )

    print("\nConstrained TT, TGW, GWGW...")
    compute_super_S(
        out_files[16],
        out_files[0],
        out_files[2],
        out_files[4],
        out_files[5],
        out_files[11],
        out_files[12],
    )

    print("\nMasked unconstrained TT, TGW, GWGW...")
    compute_super_S(
        out_files[17],
        out_files[1],
        out_files[1],
        out_files[4],
        out_files[4],
        out_files[13],
        out_files[13],
    )

    print("\nMasked constrained TT, TGW, GWGW...")
    compute_super_S(
        out_files[18],
        out_files[1],
        out_files[3],
        out_files[4],
        out_files[6],
        out_files[13],
        out_files[14],
    )
    return


def main(CurrentState: State):
    save_single_estimators(CurrentState)
    save_combined_estimators(CurrentState.settings.get_S_files())
    return


if __name__ == "__main__":
    main()
