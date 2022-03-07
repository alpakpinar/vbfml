import os

from glob import glob
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from matplotlib import pyplot as plt

from vbfml.training.data import TrainingLoader

pjoin = os.path.join


@dataclass
class HistorySet:
    """
    Object to retrieve and save a set of training histories from a range
    of models (i.e. from a grid search), and plot the desired training metrics.
    """

    output_directory: str
    histories: "dict" = field(default_factory=dict)

    def accumulate_histories(self, num_models: int = -1) -> None:
        """Retrieve the set of model directories and accumulate the history data for all models."""
        dirs = glob(pjoin(self.output_directory, "model_v*"))
        if num_models > 0 and num_models < len(dirs):
            dirs = dirs[:num_models]

        for model_dir in dirs:
            loader = TrainingLoader(model_dir)
            model_key = os.path.basename(model_dir.rstrip("/"))
            try:
                self.histories[model_key] = loader.get_history()
            except FileNotFoundError:
                print(f"WARNING: history.pkl not found for {model_key}, skipping.")
                continue

    def plot(self, quantity: str = "loss"):
        """For the set of models, plot the quantity."""
        if len(self.histories) == 0:
            raise RuntimeError(
                "No histories found, did you forget to call accumulate_histories()?"
            )

        fig, ax = plt.subplots()
        for key, history in self.histories.items():
            x, y = history[f"x_{quantity}"], history[f"y_{quantity}"]
            ax.plot(x, y, marker="o", label=key)

        ax.legend()
        ax.set_xlabel("Training Time (a.u.)", fontsize=14)
        ax.set_ylabel(quantity, fontsize=14)

        if "loss" in quantity:
            ax.set_yscale("log")
        elif "accuracy" in quantity:
            ax.set_ylim(0, 1.1)

        # Save figure
        outdir = pjoin(self.output_directory, "plots")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = pjoin(outdir, f"{quantity}.pdf")
        fig.savefig(outpath)
        plt.close(fig)
