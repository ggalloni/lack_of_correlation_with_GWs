import numpy as np
from classes import State
from functions import summing


def get_triangular_mask(CurrentState: State):
    size = 180 - int(180 / (CurrentState.settings.lmax - 1)) + 1
    mask = np.tri(size, k=-1, dtype=int)[:, ::-1]
    mask = np.array([mask for i in range(CurrentState.settings.N)])
    return np.swapaxes(mask, axis1=0, axis2=2)


def save_cumulative_S(CurrentState: State):
    mask = get_triangular_mask(CurrentState)
    in_files = CurrentState.settings.get_S_files()
    out_files = CurrentState.settings.get_cumu_S_files()

    print("\nSumming over angular configurations...")

    print("\nTT...")
    summing(out_files[0], in_files[0], in_files[0], mask)
    print("\nMasked TT...")
    summing(out_files[1], in_files[1], in_files[1], mask)
    print("\nSMICA...")
    summing(out_files[2], in_files[0], in_files[2], mask)
    print("\nMasked SMICA...")
    summing(out_files[3], in_files[1], in_files[3], mask)
    print("\nUnconstrained GWGW...")
    summing(out_files[4], in_files[4], in_files[4], mask)
    print("\nConstrained GWGW...")
    summing(out_files[5], in_files[4], in_files[5], mask)
    print("\nMasked constrained GWGW...")
    summing(out_files[6], in_files[4], in_files[6], mask)
    print("\nUnconstrained TT, GWGW...")
    summing(out_files[7], in_files[7], in_files[7], mask)
    print("\nConstrained TT, GWGW...")
    summing(out_files[8], in_files[7], in_files[8], mask)
    print("\nMasked unconstrained TT, GWGW...")
    summing(out_files[9], in_files[9], in_files[9], mask)
    print("\nMasked constrained TT, GWGW...")
    summing(out_files[10], in_files[9], in_files[10], mask)
    print("\nUnconstrained TGW...")
    summing(out_files[11], in_files[11], in_files[11], mask)
    print("\nConstrained TGW...")
    summing(out_files[12], in_files[11], in_files[12], mask)
    print("\nMasked unconstrained TGW...")
    summing(out_files[13], in_files[13], in_files[13], mask)
    print("\nMasked constrained TGW...")
    summing(out_files[14], in_files[13], in_files[14], mask)
    print("\nUnconstrained TT, TGW...")
    summing(out_files[19], in_files[19], in_files[19], mask)
    print("\nConstrained TT, TGW...")
    summing(out_files[20], in_files[19], in_files[20], mask)
    print("\nMasked unconstrained TT, TGW...")
    summing(out_files[21], in_files[21], in_files[21], mask)
    print("\nMasked constrained TT, TGW...")
    summing(out_files[22], in_files[21], in_files[22], mask)
    print("\nUnconstrained TT, TGW, GWGW...")
    summing(out_files[15], in_files[15], in_files[15], mask)
    print("\nConstrained TT, TGW, GWGW...")
    summing(out_files[16], in_files[15], in_files[16], mask)
    print("\nMasked unconstrained TT, TGW, GWGW...")
    summing(out_files[17], in_files[17], in_files[17], mask)
    print("\nMasked constrained TT, TGW, GWGW...")
    summing(out_files[18], in_files[17], in_files[18], mask)
    return


def main(CurrentState: State):
    save_cumulative_S(CurrentState)
    return


if __name__ == "__main__":
    main()

# ----------------------------------------------------- #
