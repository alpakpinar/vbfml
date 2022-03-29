#!/usr/bin/env python3

import os
import re
import click
import uproot
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from typing import List
from matplotlib import pyplot as plt

from vbfml.input.uproot import UprootReaderMultiFile
from vbfml.plot.util import ScoreDistributionPlotter, Quantity
from vbfml.training.util import get_total_n_events

pjoin = os.path.join


def make_output_dir(files: List[str]) -> str:
    """
    Based on the input directory, determine output path,
    create it if it doesn't exist and return the path.
    """
    outtag = os.path.basename(os.path.dirname(files[0]))
    outdir = pjoin("output", "plots_from_bucoffea", outtag)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir


def get_non_image_branches(file: str, tree_name: str = "sr_vbf") -> List[str]:
    """
    Given a ROOT file, peeks at branches and returns the list
    of branches which contain all branches, except the images.
    """
    f = uproot.open(file)
    all_branches = f[tree_name].keys()
    branches = [branch for branch in all_branches if not re.match("JetImage.*", branch)]
    return branches


@click.group()
@click.argument("input_files")
@click.pass_context
def cli(ctx, input_files):
    """
    input_files is the pattern for the input ROOT files, which
    also supports wildcards (*).
    """
    ctx.ensure_object(dict)
    ctx.obj["INPUT_FILES"] = input_files


@cli.command()
@click.pass_context
def scores(ctx) -> None:
    """
    Plot score distributions for the given input ROOT files.
    """
    input_files = ctx.obj["INPUT_FILES"]
    files = glob(input_files)
    assert len(files) > 0, f"No ROOT files found for pattern: {input_files}"

    outdir = make_output_dir(files)

    branches = ["score_0", "score_1"]

    reader = UprootReaderMultiFile(
        files=files,
        branches=branches,
        treename="sr_vbf",
    )

    n_events = get_total_n_events(files, "sr_vbf")
    df = reader.read_events(0, n_events)

    scores = df.to_numpy()

    # Plot the scores on an overlayed plot
    plotter = ScoreDistributionPlotter(save_to_dir=outdir)
    plotter.plot(
        scores,
        score_index=0,
        score_label="Background-like Probability",
        n_bins=20,
        left_label="MET_2017C",
    )


@cli.command()
@click.pass_context
def features(ctx) -> None:
    """
    Plot feature distributions for the given input ROOT files.
    Plots features in an overlay plot for two cases:
    1. Background prediction by model
    2. Signal prediction by model

    input_files is the pattern for the input ROOT files, which
    also supports wildcards (*).
    """
    input_files = ctx.obj["INPUT_FILES"]
    files = glob(input_files)
    assert len(files) > 0, f"No ROOT files found for pattern: {input_files}"

    outdir = make_output_dir(files)

    # Quantities to plot
    quantities = [
        "mjj",
        "detajj",
        "leadak4_eta",
        "trailak4_eta",
    ]

    reader = UprootReaderMultiFile(
        files=files,
        branches=get_non_image_branches(files[0]),
        treename="sr_vbf",
    )

    n_events = get_total_n_events(files, "sr_vbf")
    df = reader.read_events(0, n_events)

    signal_mask = df["score_1"] > df["score_0"]
    masks = {
        "Signal": signal_mask,
        "Background": ~signal_mask,
    }

    for quantity in tqdm(quantities, desc="Plotting quantities"):
        assert quantity in df, f"{quantity} not found in dataframe!"
        data = df[quantity]
        q = Quantity(name=quantity)

        fig, ax = plt.subplots()

        for label, mask in masks.items():
            ax.hist(
                data[mask],
                bins=q.bins,
                label=label,
                histtype="step",
                density=True,
            )

        ax.set_xlabel(q.label)
        ax.set_ylabel("Normalized Counts")
        ax.legend()

        outpath = pjoin(outdir, f"{quantity}.pdf")
        fig.savefig(outpath)
        plt.close(fig)


if __name__ == "__main__":
    cli()
