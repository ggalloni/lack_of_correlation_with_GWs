from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap


@dataclass()
class Settings:
    lmax: int
    savefig: bool
    show: bool
    N: int
    lmaxes: list = field(init=False)
    nside: int = field(default=64)
    smoothing: int = field(default=2)
    prefix: str = field(init=False)
    batch: str = field(default="")
    load: bool = field(default=False)
    debug: bool = field(default=False)
    cmap: ListedColormap = field(init=False)
    step: int = field(default=1)
    start: int = field(init=False)
    μ_min: np.ndarray = field(init=False)
    μ_max: np.ndarray = field(init=False)
    nbins: int = field(default=50)
    normalize: bool = field(default=True)

    def __post_init__(self) -> None:
        self.set_directories()
        self.lmaxes = [4, 6, 10]
        self.prefix = f"lmax{self.lmax}_nside{self.nside}_"
        self.start = int(180 / (self.lmax - 1))
        self.μ_min = np.arange(self.start, 180 + self.step, self.step)
        self.μ_max = np.arange(self.start, 180 + self.step, self.step)
        self.get_colormap()

    def set_directories(self) -> None:
        package_path = Path(__file__).parent
        data_path = package_path.joinpath("data")
        data_path.mkdir(parents=True, exist_ok=True)
        self.dir = data_path.joinpath(self.batch)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.dir.joinpath("figures")
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.I_file = self.dir.joinpath(f"lmax{self.lmax}_I.pkl")
        self.P_file = self.dir.joinpath(f"lmax{self.lmax}_P.pkl")
        self.CLS_file = self.dir.joinpath("../CLS.pkl")
        self.opt_angs_file = self.dir.joinpath(f"lmax{self.lmax}_opt_angs.pkl")
        return

    def get_colormap(self) -> None:
        self.cmap = ListedColormap(
            np.loadtxt(self.dir.joinpath("../../Planck_data/Planck_cmap.txt")) / 255.0
        )
        self.cmap.set_bad("gray")  # color of missing pixels
        self.cmap.set_under("white")

    def get_CLs(self) -> dict:
        with open(self.CLS_file, "rb") as pickle_file:
            CLs = pickle.load(pickle_file)
        cons_0 = CLs["gwgw"] - CLs["tgw"] ** 2 / CLs["tt"]
        cons_0[np.isnan(cons_0)] = 0.0
        CLs["constrained_gwgw"] = cons_0
        return CLs

    def get_mask(self) -> np.ndarray:
        from healpy import read_map

        from functions import downgrade_map

        mask_file = self.dir.joinpath(
            "../../Planck_data/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
        )
        mask = read_map(mask_file)
        mask = downgrade_map(mask, self.nside, fwhmout=np.deg2rad(self.smoothing))

        thrshld = 0.9
        mask[mask < thrshld] = 0
        mask[mask >= thrshld] = 1

        return mask

    def get_cmb_map(self) -> np.ndarray:
        from healpy import read_map

        from functions import downgrade_map

        planck_filename = self.dir.joinpath(
            "../../Planck_data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
        )
        planck_map = read_map(planck_filename, field=0) * 1e6  # We want muK2
        return downgrade_map(planck_map, self.nside, fwhmout=np.deg2rad(self.smoothing))

    def get_spectra_files(self):
        return [
            self.dir.joinpath(self.prefix + "TT_uncon_CL.pkl"),  # 0
            self.dir.joinpath(self.prefix + "GWGW_uncon_CL.pkl"),  # 1
            self.dir.joinpath(self.prefix + "masked_TT_uncon_CL.pkl"),  # 2
            self.dir.joinpath(self.prefix + "GWGW_con_CL.pkl"),  # 3
            self.dir.joinpath(self.prefix + "masked_GWGW_con_CL.pkl"),  # 4
            self.dir.joinpath(self.prefix + "TGW_uncon_CL.pkl"),  # 5
            self.dir.joinpath(self.prefix + "TGW_con_CL.pkl"),  # 6
            self.dir.joinpath(self.prefix + "masked_TGW_uncon_CL.pkl"),  # 7
            self.dir.joinpath(self.prefix + "masked_TGW_con_CL.pkl"),  # 8
        ]

    def get_S_files(self):
        return [
            self.dir.joinpath(self.prefix + "S_TT.pkl"),  # 0
            self.dir.joinpath(self.prefix + "S_masked_TT.pkl"),  # 1
            self.dir.joinpath(self.prefix + "S_SMICA.pkl"),  # 2
            self.dir.joinpath(self.prefix + "S_masked_SMICA.pkl"),  # 3
            self.dir.joinpath(self.prefix + "S_uncon_GWGW.pkl"),  # 4
            self.dir.joinpath(self.prefix + "S_con_GWGW.pkl"),  # 5
            self.dir.joinpath(self.prefix + "S_masked_con_GWGW.pkl"),  # 6
            self.dir.joinpath(self.prefix + "S_uncon_TT_GWGW.pkl"),  # 7
            self.dir.joinpath(self.prefix + "S_con_TT_GWGW.pkl"),  # 8
            self.dir.joinpath(self.prefix + "S_masked_uncon_TT_GWGW.pkl"),  # 9
            self.dir.joinpath(self.prefix + "S_masked_con_TT_GWGW.pkl"),  # 10
            self.dir.joinpath(self.prefix + "S_uncon_TGW.pkl"),  # 11
            self.dir.joinpath(self.prefix + "S_con_TGW.pkl"),  # 12
            self.dir.joinpath(self.prefix + "S_masked_uncon_TGW.pkl"),  # 13
            self.dir.joinpath(self.prefix + "S_masked_con_TGW.pkl"),  # 14
            self.dir.joinpath(self.prefix + "S_uncon_TT_TGW_GWGW.pkl"),  # 15
            self.dir.joinpath(self.prefix + "S_con_TT_TGW_GWGW.pkl"),  # 16
            self.dir.joinpath(self.prefix + "S_masked_uncon_TT_TGW_GWGW.pkl"),  # 17
            self.dir.joinpath(self.prefix + "S_masked_con_TT_TGW_GWGW.pkl"),  # 18
            self.dir.joinpath(self.prefix + "S_uncon_TT_TGW.pkl"),  # 19
            self.dir.joinpath(self.prefix + "S_con_TT_TGW.pkl"),  # 20
            self.dir.joinpath(self.prefix + "S_masked_uncon_TT_TGW.pkl"),  # 21
            self.dir.joinpath(self.prefix + "S_masked_con_TT_TGW.pkl"),  # 22
        ]

    def get_cumu_S_files(self):
        return [
            self.dir.joinpath(self.prefix + "cumu_S_TT.pkl"),  # 0
            self.dir.joinpath(self.prefix + "cumu_S_masked_TT.pkl"),  # 1
            self.dir.joinpath(self.prefix + "cumu_S_SMICA.pkl"),  # 2
            self.dir.joinpath(self.prefix + "cumu_S_masked_SMICA.pkl"),  # 3
            self.dir.joinpath(self.prefix + "cumu_S_uncon_GWGW.pkl"),  # 4
            self.dir.joinpath(self.prefix + "cumu_S_con_GWGW.pkl"),  # 5
            self.dir.joinpath(self.prefix + "cumu_S_masked_con_GWGW.pkl"),  # 6
            self.dir.joinpath(self.prefix + "cumu_S_uncon_TT_GWGW.pkl"),  # 7
            self.dir.joinpath(self.prefix + "cumu_S_con_TT_GWGW.pkl"),  # 8
            self.dir.joinpath(self.prefix + "cumu_S_masked_uncon_TT_GWGW.pkl"),  # 9
            self.dir.joinpath(self.prefix + "cumu_S_masked_con_TT_GWGW.pkl"),  # 10
            self.dir.joinpath(self.prefix + "cumu_S_uncon_TGW.pkl"),  # 11
            self.dir.joinpath(self.prefix + "cumu_S_con_TGW.pkl"),  # 12
            self.dir.joinpath(self.prefix + "cumu_S_masked_uncon_TGW.pkl"),  # 13
            self.dir.joinpath(self.prefix + "cumu_S_masked_con_TGW.pkl"),  # 14
            self.dir.joinpath(self.prefix + "cumu_S_uncon_TT_TGW_GWGW.pkl"),  # 15
            self.dir.joinpath(self.prefix + "cumu_S_con_TT_TGW_GWGW.pkl"),  # 16
            self.dir.joinpath(
                self.prefix + "cumu_S_masked_uncon_TT_TGW_GWGW.pkl"
            ),  # 17
            self.dir.joinpath(self.prefix + "cumu_S_masked_con_TT_TGW_GWGW.pkl"),  # 18
            self.dir.joinpath(self.prefix + "cumu_S_uncon_TT_TGW.pkl"),  # 19
            self.dir.joinpath(self.prefix + "cumu_S_con_TT_TGW.pkl"),  # 20
            self.dir.joinpath(self.prefix + "cumu_S_masked_uncon_TT_TGW.pkl"),  # 21
            self.dir.joinpath(self.prefix + "cumu_S_masked_con_TT_TGW.pkl"),  # 22
        ]

    def get_significance_files(self):
        return [
            self.dir.joinpath(self.prefix + "signi_TT.pkl"),  # 0
            self.dir.joinpath(self.prefix + "signi_masked_TT.pkl"),  # 1
            self.dir.joinpath(self.prefix + "signi_SMICA.pkl"),  # 2
            self.dir.joinpath(self.prefix + "signi_masked_SMICA.pkl"),  # 3
            self.dir.joinpath(self.prefix + "signi_uncon_GWGW.pkl"),  # 4
            self.dir.joinpath(self.prefix + "signi_con_GWGW.pkl"),  # 5
            self.dir.joinpath(self.prefix + "signi_masked_con_GWGW.pkl"),  # 6
            self.dir.joinpath(self.prefix + "signi_uncon_TT_GWGW.pkl"),  # 7
            self.dir.joinpath(self.prefix + "signi_con_TT_GWGW.pkl"),  # 8
            self.dir.joinpath(self.prefix + "signi_masked_uncon_TT_GWGW.pkl"),  # 9
            self.dir.joinpath(self.prefix + "signi_masked_con_TT_GWGW.pkl"),  # 10
            self.dir.joinpath(self.prefix + "signi_uncon_TGW.pkl"),  # 11
            self.dir.joinpath(self.prefix + "signi_con_TGW.pkl"),  # 12
            self.dir.joinpath(self.prefix + "signi_masked_uncon_TGW.pkl"),  # 13
            self.dir.joinpath(self.prefix + "signi_masked_con_TGW.pkl"),  # 14
            self.dir.joinpath(self.prefix + "signi_uncon_TT_TGW_GWGW.pkl"),  # 15
            self.dir.joinpath(self.prefix + "signi_con_TT_TGW_GWGW.pkl"),  # 16
            self.dir.joinpath(self.prefix + "signi_masked_uncon_TT_TGW_GWGW.pkl"),  # 17
            self.dir.joinpath(self.prefix + "signi_masked_con_TT_TGW_GWGW.pkl"),  # 18
            self.dir.joinpath(self.prefix + "signi_uncon_TT_TGW.pkl"),  # 19
            self.dir.joinpath(self.prefix + "signi_con_TT_TGW.pkl"),  # 20
            self.dir.joinpath(self.prefix + "signi_masked_uncon_TT_TGW.pkl"),  # 21
            self.dir.joinpath(self.prefix + "signi_masked_con_TT_TGW.pkl"),  # 22
        ]


@dataclass
class State:
    settings: Settings
    signi_dict: dict = field(init=False)
    opt_angs: dict = field(init=False)
    spectra_files: list[str] = field(init=False)
    S_files: list[str] = field(init=False)
    cumu_S_files: list[str] = field(init=False)
    CLs: dict = field(init=False)
    cmb_map: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.signi_dict = {}
        self.save_CLs()
        self.save_cmb_map()
        self.save_mask()
        self.save_widgets()
        self.save_SMICA_spectra()
        self.window_file = self.settings.dir.joinpath(
            self.settings.prefix + "window.pkl"
        )

    def save_cmb_map(self) -> None:
        cmb_map = self.settings.get_cmb_map()
        self.cmb_map = cmb_map

    def save_mask(self) -> None:
        mask = self.settings.get_mask()
        self.mask = mask

    def save_CLs(self) -> None:
        CLs = self.settings.get_CLs()
        self.CLs = CLs

    def save_widgets(self) -> None:
        from progressbar import (
            AdaptiveETA,
            AnimatedMarker,
            Bar,
            FileTransferSpeed,
            Percentage,
        )

        self.widgets = [
            "DUMMY: ",
            Percentage(),
            " ",
            AnimatedMarker(markers="←↖↑↗→↘↓↙"),
            Bar(marker="0", left="[", right="]"),
            " ",
            AdaptiveETA(),
            " ",
            FileTransferSpeed(),
        ]
        return

    def get_window(self) -> np.ndarray:
        file = self.window_file
        with open(file, "rb") as pickle_file:
            return pickle.load(pickle_file)

    def get_legendre_integrals(self) -> np.ndarray:
        file = self.settings.I_file
        with open(file, "rb") as pickle_file:
            return pickle.load(pickle_file)

    def collect_spectra(self, files: list[str]) -> dict:
        names = [(file.stem).replace(self.settings.prefix, "") for file in files]

        data = {}

        for name, file in zip(names, files):
            with open(file, "rb") as f:
                array = pickle.load(f)
                data[name] = array

        return data

    def save_SMICA_spectra(self) -> None:
        from healpy import alm2cl, map2alm
        from pymaster import NmtBin, NmtField, NmtWorkspace, compute_full_master

        lmax = self.settings.lmax
        nside = self.settings.nside

        alm_planck = map2alm(self.cmb_map, lmax=lmax, use_pixel_weights=True)
        planck_CL = alm2cl(alm_planck)
        f_T = NmtField(self.mask, [self.cmb_map])
        b = NmtBin.from_nside_linear(nside, 1)
        wsp = NmtWorkspace()
        wsp.compute_coupling_matrix(f_T, f_T, b, is_teb=False)
        planck_masked_CL = np.insert(
            compute_full_master(f_T, f_T, b, workspace=wsp)[0][: lmax + 1],
            [0, 0],
            [0, 0],
        )

        planck_CL = np.swapaxes(planck_CL[:, np.newaxis], axis1=0, axis2=1)
        planck_masked_CL = np.swapaxes(
            planck_masked_CL[:, np.newaxis], axis1=0, axis2=1
        )
        self.SMICA_Cl = planck_CL
        self.masked_SMICA_Cl = planck_masked_CL
        return
