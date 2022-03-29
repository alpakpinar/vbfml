#!/usr/bin/env python3

import os
import re
import click
import uproot
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt

from vbfml.input.uproot import UprootReaderMultiFile
from vbfml.plot.util import ScoreDistributionPlotter

pjoin = os.path.join


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_files")
def scores(input_files: str) -> None:
    """
    Plot score distributions for the given input ROOT files.

    input_files is the pattern for the input ROOT files, which
    also supports wildcards (*).
    """
    files = glob(input_files)
    assert len(files) > 0, f"No ROOT files found for pattern: {input_files}"

    outtag = os.path.basename(os.path.dirname(files[0]))
    outdir = pjoin("output", "plots_from_bucoffea", outtag)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    branches = ["score_0", "score_1"]

    reader = UprootReaderMultiFile(
        files=files,
        branches=branches,
        treename="sr_vbf",
    )

    df = reader.read_events(0, int(1e5))

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


if __name__ == "__main__":
    cli()
