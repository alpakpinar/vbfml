#!/usr/bin/env python3

import os
import re
import uproot
import click
import numpy as np
import awkward as ak

from vbfml.training.plot import ImagePlotter
from vbfml.util import get_process_tag_from_file

pjoin = os.path.join


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-i",
    "--input-file",
    required=True,
    help="Path to the ROOT file.",
)
@click.option(
    "-b",
    "--image-branch",
    required=False,
    default="JetImageFine_E",
    help="Name of the branch containing image data.",
)
@click.option(
    "--num-events",
    default=10000,
    type=int,
    required=False,
    help="Number of events to take the average of.",
)
def plot_average(input_file: str, image_branch: str, num_events: int):
    """
    Plots the averaged image for the given branch.
    """
    tree = uproot.open(input_file)["sr_vbf"]

    table_name = image_branch.split('_')[0]
    n_eta_bins = int( tree[f'{table_name}_nEtaBins'].array()[0] )
    n_phi_bins = int( tree[f'{table_name}_nPhiBins'].array()[0] )

    num_entries = len(tree[f'{table_name}_nEtaBins'].array())
    num_events = min(num_events, num_entries)

    # Read out the image and compute the average
    arr = ak.to_numpy(tree[f"{image_branch}_preprocessed"])[:num_events]
    mean_arr = np.mean(arr, axis=0)

    plotter = ImagePlotter(n_eta_bins=n_eta_bins, n_phi_bins=n_phi_bins)
    plotter.plot(
        mean_arr,
        outdir="./test",
        filename="test.pdf",
        vmin=1e-1,
        vmax=255,
    )


if __name__ == "__main__":
    cli()
