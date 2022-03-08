import os

from glob import glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from collections import OrderedDict

from vbfml.training.data import TrainingLoader

pjoin = os.path.join


@dataclass
class HistorySet:
    """
    Object to retrieve and save a set of training histories from a range
    of models (i.e. from a grid search), and plot the desired training metrics.
    """

    output_directory: str
    histories: "dict" = field(default_factory=OrderedDict)

    def __len__(self):
        """Returns the number of histories that this object holds."""
        return len(self.histories)

    def accumulate_histories(self) -> None:
        """Retrieve the set of model directories and accumulate the history data for all models."""
        dirs = glob(pjoin(self.output_directory, "model_v*"))

        for model_dir in dirs:
            loader = TrainingLoader(model_dir)
            model_key = os.path.basename(model_dir.rstrip("/"))
            try:
                self.histories[model_key] = loader.get_history()
            except FileNotFoundError:
                print(f"WARNING: history.pkl not found for {model_key}, skipping.")
                continue

    def plot(
        self, quantity: str = "loss", model_range: Tuple[int, int] = (0, -1)
    ) -> None:
        """
        For the set of models, plot the training metric (quantity).
        model_range argument can be specified if only a subset of models
        are to be plotted on the same plot.
        """
        if len(self) == 0:
            raise RuntimeError(
                "No histories found, did you forget to call accumulate_histories()?"
            )

        model_range_start, model_range_end = model_range
        if model_range_end > 0 and model_range_end < len(dirs):
            histories = list(self.histories.items())[model_range_start:model_range_end]
            histories = dict(histories)

        fig, ax = plt.subplots()
        for key, history in self.histories.items():
            x, y = history[f"x_{quantity}"], history[f"y_{quantity}"]
            ax.plot(x, y, marker="o", label=key)

        ax.legend(ncol=(len(self.histories) - 1) // 5 + 1)
        ax.set_xlabel("Training Time (a.u.)", fontsize=14)
        ax.set_ylabel(quantity, fontsize=14)

        if "loss" in quantity:
            ax.set_yscale("log")
            ax.set_ylim(3e-5, 8e-5)
        elif "accuracy" in quantity:
            ax.set_ylim(0, 1.1)

        # Save figure
        outdir = pjoin(self.output_directory, "plots")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = pjoin(outdir, f"{quantity}.pdf")
        fig.savefig(outpath)
        plt.close(fig)
