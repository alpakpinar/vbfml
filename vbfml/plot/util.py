import numpy as np

from dataclasses import dataclass


@dataclass
class Quantity:
    """
    Class to hold binning and labeling information for a given quantity.
    >>> quantity = Quantity(name)
    >>> # Access bins and labels
    >>> quantity.bins
    >>> quantity.labels
    """

    name: str

    def _set_bins(self) -> None:
        pt_bins = np.logspace(2, 3, 20)
        jet_pt_bins = np.linspace(0, 1000)
        eta_bins = np.linspace(-5, 5, 50)
        phi_bins = np.linspace(-np.pi, np.pi, 30)
        axis_bins = {
            "mjj": [
                200.0,
                400.0,
                600.0,
                900.0,
                1200.0,
                1500.0,
                2000.0,
                2750.0,
                3500.0,
                5000.0,
                8000.0,
            ],
            "detajj": np.linspace(1, 10, 20),
            "recoil_pt": pt_bins,
            "recoil_phi": phi_bins,
            "njet": np.arange(0, 10),
            "leadak4_pt": pt_bins,
            "trailak4_pt": pt_bins,
            "leadak4_eta": eta_bins,
            "trailak4_eta": eta_bins,
            "leadak4_phi": phi_bins,
            "trailak4_phi": phi_bins,
        }
        try:
            self.bins = axis_bins[self.name]
        except KeyError:
            raise RuntimeError(f"Could not find binning for quantity: {self.name}")

    def _set_label(self) -> None:
        labels = {
            "mjj": r"$M_{jj} \ (GeV)$",
            "detajj": r"$\Delta\eta_{jj}$",
            "leadak4_pt": r"Leading Jet $p_T \ (GeV)$",
            "trailak4_pt": r"Trailing Jet $p_T \ (GeV)$",
            "leadak4_eta": r"Leading Jet $\eta$",
            "trailak4_eta": r"Trailing Jet $\eta$",
            "leadak4_phi": r"Leading Jet $\phi$",
            "trailak4_phi": r"Trailing Jet $\phi$",
            "recoil_pt": r"Recoil $p_T \ (GeV)$",
            "recoil_phi": r"Recoil $\phi$",
            "njet": "Number of Jets",
        }
        try:
            self.label = labels[self.name]
        except KeyError:
            raise RuntimeError(f"Could not find the label for quantity: {self.name}")

    def __post_init__(self) -> None:
        self._set_bins()
        self._set_label()
