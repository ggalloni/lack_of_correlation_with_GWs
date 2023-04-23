from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap


@dataclass()
class Settings:
    """
    Class that contains all the settings of the project. The default values are set
     in the __post_init__ method, where the directories are also set. The __post_init__
     method is called after the __init__ method, and it is the place where the
     attributes that are not set by the user are set. The attributes that are not set
     in the __post_init__ method should be set by the user.

    Attributes
    ----------
    dir : Path
        Path to the directory where the data is stored.
    fig_dir : Path
        Path to the directory where the figures are stored.
    I_file : Path
        Path to the file where the intensity map is stored.
    P_file : Path
        Path to the file where the polarization map is stored.
    CLS_file : Path
        Path to the file where the CMB power spectra are stored.
    opt_angs_file : Path
        Path to the file where the optimal angles are stored.
    lmax : int
        Maximum multipole to be considered.
    savefig : bool
        Whether to save the figures or not.
    show : bool
        Whether to show the figures or not.
    N : int
        Number of simulated maps to be generated.
    lmaxes : list
        List of lmaxes to be considered.
    nside : int
        Nside parameter of healpy.
    smoothing : int
        Gaussian smoothing to be applied to the maps.
    prefix : str
        Prefix for the files.
    batch : str
        Name of the folder where the data is stored.
    load : bool
        Whether to load the data or not.
    debug : bool
        Whether to print the messages or not.
    cmap : ListedColormap
        Colormap for the figures.
    step : int
        Step for the binning of the optimal angles.
    start : int
        Start of the binning of the optimal angles.
    μ_min : np.ndarray
        Minimum values of the bins for the optimal angles.
    μ_max : np.ndarray
        Maximum values of the bins for the optimal angles.
    nbins : int
        Number of bins for the optimal angles.
    normalize : bool
        Whether to normalize the histograms or not.
    """

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
    dir: Path = field(init=False)
    fig_dir: Path = field(init=False)
    I_file: Path = field(init=False)
    P_file: Path = field(init=False)
    CLS_file: Path = field(init=False)
    opt_angs_file: Path = field(init=False)

    def __post_init__(self) -> None:
        """
        Set the directories, the lmaxes, the prefix, the start of the binning of the
        optimal angles, the minimum and maximum values of the bins for the optimal
        angles, and the colormap.
        """
        self.set_directories()
        self.lmaxes = [4, 6, 10]
        self.prefix = f"lmax{self.lmax}_nside{self.nside}_"
        self.start = int(180 / (self.lmax - 1))
        self.μ_min = np.arange(self.start, 180 + self.step, self.step)
        self.μ_max = np.arange(self.start, 180 + self.step, self.step)
        self.get_colormap()

    def set_directories(self) -> None:
        """
        Set the directories where the data is stored, the figures are stored, the
        intensity map is stored, the polarization map is stored, the CMB power
        spectra are stored, and the optimal angles are stored. Note that it will create missing directories.
        """
        package_path = Path(__file__).parent
        data_path = package_path.joinpath("../../data")
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
        """
        Set the colormap for the figures.
        """
        self.cmap = ListedColormap(
            np.loadtxt(self.dir.joinpath("../../Planck_data/Planck_cmap.txt")) / 255.0
        )
        self.cmap.set_bad("gray")
        self.cmap.set_under("white")

    def get_CLs(self) -> dict:
        """Get the CMB power spectra from the file where they are stored.

        Returns
        -------
        CLs : dict
            Dictionary with the CMB and the CGWB power spectra.
        """
        with open(self.CLS_file, "rb") as pickle_file:
            CLs = pickle.load(pickle_file)
        constrained_gwgw = CLs["gwgw"] - CLs["tgw"] ** 2 / CLs["tt"]
        constrained_gwgw[np.isnan(constrained_gwgw)] = 0.0
        CLs["constrained_gwgw"] = constrained_gwgw
        return CLs

    def get_mask(self) -> np.ndarray:
        """
        Get the mask from the file where it is stored.

        Returns
        -------
        mask : np.ndarray
            Mask for the CMB map.
        """
        from healpy import read_map

        from GW_lack_corr.src.common_functions import downgrade_map

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
        """
        Get the CMB map from the file where it is stored.

        Returns
        -------
        CMB_map : np.ndarray
            CMB map.
        """
        from healpy import read_map

        from GW_lack_corr.src.common_functions import downgrade_map

        planck_filename = self.dir.joinpath(
            "../../Planck_data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
        )
        planck_map = read_map(planck_filename, field=0) * 1e6  # We want muK2
        return downgrade_map(planck_map, self.nside, fwhmout=np.deg2rad(self.smoothing))

    def get_spectra_files(self) -> list:
        """
        Get the files where the spectra are stored.

        Returns
        -------
        files : list of Path
            List of the files where the spectra are stored.
        """
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

    def get_S_files(self) -> list:
        """
        Get the files where the S estimators are stored.

        Returns
        -------
        files : list of Path
            List of the files where the S estimators are stored.
        """
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

    def get_cumu_S_files(self) -> list:
        """
        Get the files where the cumulative S estimators are stored.

        Returns
        -------
        files : list of Path
            List of the files where the cumulative S estimators are stored.
        """
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

    def get_significance_files(self) -> list:
        """
        Get the files where the significance estimators are stored.

        Returns
        -------
        files : list of Path
            List of the files where the significances are stored.
        """
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
