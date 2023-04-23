from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

from GW_lack_corr.classes.settings import Settings


@dataclass
class State:
    """
    This class stores the state of the application. It is used to store the
    settings, the data and the results. It is used to save and load the state
    of the application. This allows accessing useful information from different parts of the code.

    Attributes
    ----------
    settings: Settings
        The settings of the application.
    signi_dict: dict
        The significance of the different fields.
    opt_angs: dict
        The optimal angles of the different fields.
    spectra_files: list
        The filenames of the different spectra.
    S_files: list
        The filenames of the different S estimators.
    cumu_S_files: list
        The filenames of the different cumulative S estimators.
    CLs: dict
        The different CMB power spectra.
    cmb_map: np.ndarray
        The CMB map.
    mask: np.ndarray
        The mask of the CMB map.
    window_file: Path
        The filename of the window function.
    """

    settings: Settings
    signi_dict: dict = field(init=False)
    opt_angs: dict = field(init=False)
    spectra_files: list = field(init=False)
    S_files: list = field(init=False)
    cumu_S_files: list = field(init=False)
    CLs: dict = field(init=False)
    cmb_map: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        This method is called after the __init__ method. It initializes the
        attributes of the class.
        """
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
        """
        This method saves the cmb map.
        """
        cmb_map = self.settings.get_cmb_map()
        self.cmb_map = cmb_map

    def save_mask(self) -> None:
        """
        This method saves the mask.
        """
        mask = self.settings.get_mask()
        self.mask = mask

    def save_CLs(self) -> None:
        """
        This method saves the power spectra.
        """
        CLs = self.settings.get_CLs()
        self.CLs = CLs

    def save_widgets(self) -> None:
        """
        This method initializes the widgets necessary to plot the progessbars.
        """
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
        """
        This method returns the window function.

        Returns
        -------
        window: np.ndarray
            The window function of the considered mask.
        """
        file = self.window_file
        with open(file, "rb") as pickle_file:
            return pickle.load(pickle_file)

    def get_legendre_integrals(self) -> np.ndarray:
        """
        This method returns the legendre integrals.

        Returns
        -------
        legendre_integrals: np.ndarray
            The legendre integrals.
        """
        file = self.settings.I_file
        with open(file, "rb") as pickle_file:
            return pickle.load(pickle_file)

    def collect_spectra(self, files: list) -> dict:
        """
        This method collects the different spectra given a list of filenames.

        Parameters
        ----------
        files: list
            The filenames of the different spectra.

        Returns
        -------
        data: dict
            The different spectra.
        """
        names = [(file.stem).replace(self.settings.prefix, "") for file in files]

        data = {}

        for name, file in zip(names, files):
            with open(file, "rb") as f:
                array = pickle.load(f)
                data[name] = array

        return data

    def save_SMICA_spectra(self) -> None:
        """
        This method saves the spectra of the SMICA CMB map. This is done both for the full-sky case, with anafast, and for the masked case, with the pseudo-Cl algorithm.
        """
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
