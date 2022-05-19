import os
import numpy as np
import pandas as pd

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
            "njet_pt30": np.arange(0, 10),
            "njet_central": np.arange(0, 10),
            "njet_forward": np.arange(0, 10),
            "leadak4_pt": pt_bins,
            "trailak4_pt": pt_bins,
            "leadak4_eta": eta_bins,
            "trailak4_eta": eta_bins,
            "leadak4_phi": phi_bins,
            "trailak4_phi": phi_bins,
            "dphi_ak40_met": phi_bins,
            "dphi_ak41_met": phi_bins,
            "minDPhiJetMet": phi_bins,
            "ak4_pt2": np.linspace(0, 250, 26),
            "ak4_eta2": eta_bins,
            "ak4_phi2": phi_bins,
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
            "njet_pt30": r"Number of Jets With $p_T > 30 \ GeV$",
            "njet_central": "Number of Central Jets",
            "njet_forward": "Number of Forward Jets",
            "dphi_ak40_met": r"$\Delta\phi$(leading jet, MET)",
            "dphi_ak41_met": r"$\Delta\phi$(trailing jet, MET)",
            "minDPhiJetMet": r"min$\Delta\phi$(jet, MET)",
            "ak4_pt2": r"Third Jet $p_T \ (GeV)$",
            "ak4_eta2": r"Third Jet $\eta$",
            "ak4_phi2": r"Third Jet $\phi$",
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
        weights: np.ndarray,
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
        ax.hist(scores_for_i, weights=weights, bins=n_bins, histtype="step")

        ax.set_xlabel(score_label, fontsize=14)
        ax.set_ylabel("Weighted Counts", fontsize=14)

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


def plot_histograms_for_each_label(
    data: pd.DataFrame,
    variable: str,
    cut: float,
    outdir: str,
) -> None:
    """
    plot the distribution of a feature according to its label (signal or bkg) from a pandas data frame with a "label" column
    specify the dataframe and the string of the head of the variable's column you want to plot
    Is also draw the line of a speficic cut to see how it would differentiate bkg from signal
    """

    plt.figure()
    plt.hist(
        data[data["labels"] == 0][variable],
        bins=50,
        density=True,
        weights=data[data["labels"] == 0]["weights"],
        histtype="step",
        label="background",
    )
    plt.hist(
        data[data["labels"] == 1][variable],
        bins=50,
        density=True,
        weights=data[data["labels"] == 1]["weights"],
        histtype="step",
        label="signal",
    )
    plt.axvline(x=cut, color="k", linestyle="--", label=f"cut at {cut:.3f}")
    plt.title(f"{variable} distribution")
    plt.xlabel(variable)
    plt.ylabel("density counts")
    plt.legend()
    plt.savefig(pjoin(outdir, f"{variable}_density.pdf"))


def sort_by_pair_along_first_axis(
    x: list,
    y: list,
    z: list = [None],
    reverse: bool = False,
    abs_val: bool = False,
):
    """
    take 2(3) lists/np.array of the same length representing x-y(-z) coordinate-like objects, y_i(-z_i) need to stay with x_i
    sort them in the good order according to x from lowest to highest (reverse=True for high to low)
    return 2(3) np.arrays with x sorted, y(-z) got the same permutation as x
    /!/ sort by absolute values !
    """

    # in case of 2 lists, create a dummy third one
    z_flag = True
    if z[0] == None:
        z_flag = False
        z = [0] * len(x)

    # create list of coupled elements
    x_y_z = []
    for i in range(len(x)):
        x_y_z.append([x[i], y[i], z[i]])

    # define function used to sort the array
    def absfirst(a: list):
        return abs(a[0])

    def first(a: list):
        return a[0]

    # sort the arrays w/o absolute value
    if abs_val:
        x_y_z.sort(key=absfirst, reverse=reverse)
    else:
        x_y_z.sort(key=first, reverse=reverse)

    x_y_z = np.array(x_y_z)
    x = x_y_z[:, 0]
    y = x_y_z[:, 1]
    z = x_y_z[:, 2]

    if z_flag:
        return x, y, z
    else:
        return x, y
