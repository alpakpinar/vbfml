#!/usr/bin/env python3

import os

import click
import warnings
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from glob import glob
from pprint import pprint
from tqdm import tqdm

from vbfml.training.data import TrainingLoader
from vbfml.input.uproot import UprootReaderMultiFile

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_files")
@click.argument("training_path")
def predict(input_files: str, training_path: str):
    loader = TrainingLoader(training_path)
    model = loader.get_model()

    # Use glob to handle wildcard characters
    files = glob(input_files)

    reader = UprootReaderMultiFile(files=files, branches=None, treename="sr_vbf")

    n_events = int(1e4)
    df = reader.read_events(0, n_events)

    # Add weight column
    df["weight"] = df["weight_total"] * df["xs"] / df["sumw"]

    # Feature columns
    image_pixels = df.filter(regex="JetIm.*pixels.*").to_numpy()

    # Make predictions with the pre-trained model
    predictions = model.predict(image_pixels).argmax(axis=1)

    # Based on the predictions, we histogram different classes
    quantities_labels = {
        "mjj": r"$M_{jj} \ (GeV)$",
        "detajj": r"$\Delta \eta_{jj}$",
        "leadak4_eta": r"Leading Jet $\eta$",
        "trailak4_eta": r"Trailing Jet $\eta$",
    }

    outdir = pjoin(training_path, "plots", "predictions")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Dump the input file argument to a txt file
    input_file = pjoin(outdir, "input_root_files.txt")
    with open(input_file, "w+") as f:
        f.write(f"{input_files}\n")

    for quantity, xlabel in tqdm(quantities_labels.items(), desc="Plotting histograms"):
        fig, ax = plt.subplots()
        n_bins = 50
        for sample_cls in [0, 1]:
            mask = predictions == sample_cls
            ax.hist(
                df[quantity][mask],
                histtype="step",
                weights=df["weight"][mask],
                bins=n_bins,
                label=f"class_{sample_cls}",
            )
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel("Weighted Counts", fontsize=14)

        ax.legend()

        outpath = pjoin(outdir, f"{quantity}.pdf")
        fig.savefig(outpath)
        plt.close(fig)


if __name__ == "__main__":
    cli()
