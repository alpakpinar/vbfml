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
            "dphijj": np.linspace(0, 1.5, 10),
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
            "dphijj": r"$\Delta\phi_{jj}$",
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
    outdir: str,
    datasets: tuple = ("qcd_v", "ewk_v", "vbf_h"),
    cut: float = 0,
    save_name: str = "",
    cut_title: str = "",
) -> None:
    """
    plot the distribution of a feature according to its dataset_label (['qcd_v', 'ewk_v', 'vbf_h']) from a pandas data frame
    specify the dataframe and the string of the head of the variable's column you want to plot
    the dataframe should contains the following : 'dataset_to_read', 'dataset_label', 'weights' and the variable
    If cut specified, also draws the line of the specific cut to see how it would differentiate bkg from signal
    """

    # adjust binning
    bins = 50
    if variable == "score":
        bins = 20
    if variable == "dphijj":
        bins = 10
        # additional cut because ~0.03% of QCD have dphijj > 1.5 and mess up the binning range
        data = data[data["dphijj"] < 1.5]

    plt.figure()

    for dataset_label in datasets:
        data_label = data[data["dataset_label"] == dataset_label]

        if len(data_label) != 0:
            plt.hist(
                data_label[variable],
                bins=bins,
                density=True,
                weights=data_label["weights"],
                histtype="step",
                label=dataset_label,
            )

    if cut:
        plt.axvline(x=cut, color="k", linestyle="--", label=f"cut at {cut:.3f}")
    plt.title(f"{variable} distribution{cut_title}")
    plt.xlabel(variable)
    plt.ylabel("density counts")
    plt.ylim(bottom=0)
    if variable == "mjj":
        plt.xlim([0, 5000])
        plt.legend()
    elif variable == "dphijj":
        plt.xlim([0, 1.5])
        plt.legend(loc="lower right")
    elif variable == "score":
        plt.legend(loc="upper center")
    else:
        plt.legend()
    plt.savefig(pjoin(outdir, f"{variable}_density{cut_title}{save_name}.pdf"))


def sort_by_pair_along_first_axis(
    x: list,
    y: list,
    z: list = [None],
    reverse: bool = False,
    sort_by_abs: bool = False,
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

    # check length of lists
    assert len(x) == len(
        y
    ), f"Cannot use sort_by_pair_along_first_axis because lists (x-y) are not of same length"
    if z_flag:
        assert len(x) == len(
            y
        ), f"Cannot use sort_by_pair_along_first_axis because lists (x-y) are not of same length"

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
    if sort_by_abs:
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
