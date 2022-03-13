#!/usr/bin/env python3

import os
import re

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


def get_process_tag_from_file(filename: str) -> str:
    """
    Given a ROOT filename, return the process tag showing
    the ground truth for this process (e.g. EWK Z(vv)).
    """
    basename = os.path.basename(filename)
    mapping = {
        ".*VBF_HToInv.*M125.*": "VBF Hinv",
        ".*EWKZ2Jets.*ZToNuNu.*": "EWK Zvv",
        ".*Z\dJetsToNuNu.*PtZ.*": "QCD Zvv",
        ".*WJetsToLNu_Pt.*": "QCD Wlv",
    }

    for regex, label in mapping.items():
        if re.match(regex, basename):
            return label

    raise RuntimeError(f"Could not find a process tag for file: {basename}")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_files")
@click.argument("training_path")
@click.option(
    "-n",
    "--n-events",
    required=False,
    default=int(1e4),
    help="Number of events to process.",
)
def predict(input_files: str, training_path: str, n_events: int):
    loader = TrainingLoader(training_path)
    model = loader.get_model()

    # Use glob to handle wildcard characters
    files = glob(input_files)

    # Process tag
    process_tag = get_process_tag_from_file(files[0])
    print(f"Tag for the process   : {process_tag}")
    print(f"# of events to read   : {n_events}")

    reader = UprootReaderMultiFile(files=files, branches=None, treename="sr_vbf")

    df = reader.read_events(0, n_events)

    # Add weight column
    df["weight"] = df["weight_total"] * df["xs"] / df["sumw"]

    # Feature columns
    image_pixels = df.filter(regex="JetIm.*pixels.*").to_numpy()

    # Make predictions with the pre-trained model
    predictions = model.predict(image_pixels).argmax(axis=1)

    # Based on the predictions, we make histograms for different classes
    quantities_labels = {
        "mjj": r"$M_{jj} \ (GeV)$",
        "detajj": r"$\Delta \eta_{jj}$",
        "leadak4_eta": r"Leading Jet $\eta$",
        "trailak4_eta": r"Trailing Jet $\eta$",
    }

    outdir = pjoin(
        training_path, "plots", f"predictions_{process_tag.replace(' ', '_').lower()}"
    )
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Dump the input file argument to a txt file
    input_file = pjoin(outdir, "input_root_files.txt")
    with open(input_file, "w+") as f:
        f.write(f"{input_files}\n")

    for quantity, xlabel in tqdm(quantities_labels.items(), desc="Plotting histograms"):
        fig, ax = plt.subplots()
        n_bins = 50
        for icls, sample_cls in enumerate(["ewk_17", "v_nlo_qcd_17"]):
            mask = predictions == icls
            ax.hist(
                df[quantity][mask],
                histtype="step",
                weights=df["weight"][mask],
                bins=n_bins,
                label=sample_cls,
            )

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Weighted Counts", fontsize=14)

        ax.legend()

        ax.text(
            1,
            1,
            f"# Events: {n_events}",
            fontsize=14,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.text(
            0,
            1,
            process_tag,
            fontsize=14,
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

        outpath = pjoin(outdir, f"{quantity}.pdf")
        fig.savefig(outpath)
        plt.close(fig)


if __name__ == "__main__":
    cli()
