#!/usr/bin/env python3

import os
import re
import yaml
import click
import uproot
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from typing import List, Dict
from matplotlib import pyplot as plt

from vbfml.util import YamlLoader, vbfml_path
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


def mask_and_plot_quantity(
    quantity: Quantity, masks: Dict[str, np.array], data: pd.DataFrame, outdir: str
) -> None:
    """
    For the given set of masks, apply the mask and plot
    a histogram the masked data.
    """
    fig, ax = plt.subplots()
    for label, mask in masks.items():
        ax.hist(
            data[mask],
            bins=quantity.bins,
            label=label,
            histtype="step",
            density=True,
        )

    ax.set_xlabel(quantity.label)
    ax.set_ylabel("Normalized Counts")
    ax.legend()

    outpath = pjoin(outdir, f"{quantity.name}.pdf")
    fig.savefig(outpath)
    plt.close(fig)


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

    Plotting configurations can be updated from config/analyze/from_bucoffea.yml.
    """
    input_files = ctx.obj["INPUT_FILES"]
    files = glob(input_files)
    assert len(files) > 0, f"No ROOT files found for pattern: {input_files}"

    outdir = make_output_dir(files)

    # Read relevent information from the config file
    loader = YamlLoader(vbfml_path("config/analyze/from_bucoffea.yml"))
    config = loader.load()["scores"]

    reader = UprootReaderMultiFile(
        files=files,
        branches=config["branches"],
        treename="sr_vbf",
    )

    n_events = get_total_n_events(files, "sr_vbf")
    df = reader.read_events(0, n_events)

    scores = df[["score_0", "score_1"]].to_numpy()

    df.rename(columns={"weight_total*xs/sumw": "weight"}, inplace=True)

    # Plot the scores on an overlayed plot
    plotter = ScoreDistributionPlotter(save_to_dir=outdir)
    plot_config = config["plot"]
    plotter.plot(
        scores,
        weights=df["weight"],
        score_index=plot_config["index"],
        score_label=plot_config["label"],
        n_bins=plot_config["n_bins"],
        left_label=r"$Z(\nu\nu)$ + 2 jets",
    )


@cli.command()
@click.pass_context
def features(ctx) -> None:
    """
    Plot feature distributions for the given input ROOT files.

    Plots features in an overlay plot for separate cases, as defined
    by the masks dictionary in the script. Plotting configurations
    can be updated from config/analyze/from_bucoffea.yml.
    """
    input_files = ctx.obj["INPUT_FILES"]
    files = glob(input_files)
    assert len(files) > 0, f"No ROOT files found for pattern: {input_files}"

    outdir = make_output_dir(files)

    # Read relevent information from the config file
    loader = YamlLoader(vbfml_path("config/analyze/from_bucoffea.yml"))
    config = loader.load()["features"]

    # Quantities to plot
    quantities = config["plot"]

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
        "Very Background Like": df["score_0"] > 0.9,
        "Background Like": (df["score_0"] > 0.5) & (df["score_0"] < 0.9),
    }

    # From the config file, read which masks we want to apply and make an overlay plot
    masknames_to_run = config["masks"]["run_on"]

    masks_to_run = {}
    for maskname in masknames_to_run:
        masks_to_run[maskname] = masks[maskname].to_numpy()

    for quantity in tqdm(quantities, desc="Plotting quantities"):
        assert quantity in df, f"{quantity} not found in dataframe!"
        data = df[quantity]
        q = Quantity(name=quantity)

        mask_and_plot_quantity(q, masks=masks_to_run, data=data, outdir=outdir)


if __name__ == "__main__":
    cli()
