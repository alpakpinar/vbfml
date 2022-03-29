import os
import numpy as np

from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass
from matplotlib import pyplot as plt

pjoin = os.path.join


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
        pt_bins = np.logspace(1, 3, 20)
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


@dataclass
class ScoreDistributionPlotter:
    """
    Class for plottng the score distributions.
    """

    save_to_dir: str

    def plot(
        self,
        scores: np.ndarray,
        score_index: int,
        score_label: str,
        n_bins: int = 20,
        left_label: Optional[str] = None,
        right_label: Optional[str] = None,
    ) -> None:
        """
        Plot the score distribution among a given index.
        """
        n_classes = scores.shape[1]

        fig, ax = plt.subplots()
        scores_for_i = scores[:, score_index]
        ax.hist(scores_for_i, bins=n_bins, histtype="step")

        ax.set_xlabel(score_label, fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)

        if left_label:
            ax.text(
                0,
                1,
                left_label,
                fontsize=14,
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
        if right_label:
            ax.text(
                1,
                1,
                right_label,
                fontsize=14,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
            )

        # Save figure
        outpath = pjoin(self.save_to_dir, "score_distribution.pdf")
        fig.savefig(outpath)
        plt.close(fig)
