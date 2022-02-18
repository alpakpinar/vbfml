#!/usr/bin/env python3

import os
import sys
import re
import warnings
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
from glob import glob
from typing import Dict, Optional
from pprint import pprint

from vbfml.training.data import TrainingLoader
from vbfml.training.util import load
from vbfml.util import vbfml_path

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join


def compare_losses(
    training_paths: Dict[str, str], outdir: str, tag_regex: Optional[str] = ".*"
) -> None:
    """Retrieves and plots a comparison of loss functions for different trainings."""
    losses = {}
    metrics = [
        ("Training", "x_loss", "y_loss"),
        ("Validation", "x_val_loss", "y_val_loss"),
    ]

    markers = {
        "Training": "o",
        "Validation": "*",
    }

    def shift_by_one(xlist):
        return [x + 1 for x in xlist]

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))

    for tag, path in tqdm(training_paths.items()):
        if not re.match(tag_regex, tag):
            continue
        history = load(path)

        for metric in metrics:
            label, x_label, y_label = metric
            x, y = history[x_label], history[y_label]

            if label == "Training":
                x = shift_by_one(x)
                ax1.plot(x, y, label=tag, marker=markers[label])
            else:
                ax2.plot(x, y, label=tag, marker=markers[label])

    for ax in (ax1, ax2):
        ax.grid(True)
        ax.legend()
        ax.set_yscale("log")

    # Print the L2 regularization factor to the plot
    l2_reg_factor = re.findall("l2-(\d)", tag_regex)[0]
    ax1.text(
        1,
        1,
        f"L2 Reg. Factor: $10^{{-{l2_reg_factor}}}$",
        fontsize=14,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    ax1.set_ylabel("Training Loss", fontsize=14)
    ax2.set_ylabel("Validation Loss", fontsize=14)

    ax2.set_xlabel("Training Time (a.u.)", fontsize=14)

    outdir = pjoin(outdir, "training_comparisons")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, f"loss_comparison_{tag_regex.replace('.*','')}.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def get_history_files(training_directory: str) -> Dict[str, str]:
    """Gets the list of history*pkl files, containing training history for each model."""
    files = glob(pjoin(training_directory, "history_*.pkl"))
    files_and_tags = {}
    for f in files:
        tag = os.path.basename(f).replace("history_", "").replace(".pkl", "")
        files_and_tags[tag] = f

    return files_and_tags


def main():
    training_directory = sys.argv[1]
    files_and_tags = get_history_files(training_directory)

    for tag_regex in [f".*l2{i}.*" for i in range(-8, -5)]:
        compare_losses(files_and_tags, training_directory, tag_regex=tag_regex)


if __name__ == "__main__":
    main()
