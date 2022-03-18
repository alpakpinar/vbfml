#!/usr/bin/env python3

import os
import re
import copy
import click
import hist
import warnings
import pickle
import matplotlib.cbook
import pandas as pd
import numpy as np
import mplhep as hep
import boost_histogram as bh

from tqdm import tqdm
from matplotlib import pyplot as plt

from vbfml.training.input import build_sequence, load_datasets_bucoffea
from vbfml.training.util import (
    summarize_datasets,
    select_and_label_datasets,
    normalize_classes,
)
from vbfml.plot.util import Quantity

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join

# Dataset labels as fed for the neural network
dataset_labels_coarse = {
    "ewk_17": "(EWK.*2017|VBF_HToInvisible_M125_withDipoleRecoil_pow_pythia8_2017)",
    "v_qcd_nlo_17": "(WJetsToLNu_Pt-\d+To.*|Z\dJetsToNuNu_M-50_LHEFilterPtZ-\d+To\d+)_MatchEWPDG20-amcatnloFXFX_2017",
}

# Dataset labels per process
dataset_labels = {
    "ewkz_17": "EWKZ2Jets.*ZToNuNu.*2017",
    "ewkw_17": "EWKW(Minus|Plus)2Jets.*WToLNu.*2017",
    "qcdz_17": "Z\dJetsToNuNu.*LHEFilterPtZ.*2017",
    "qcdw_17": "WJetsToLNu_Pt.*2017",
    "vbfh_17": "VBF_HToInvisible.*M125.*2017",
}


def make_histogram(quantity_name: str) -> hist.Hist:
    """Creates an empty histogram for a given quantity."""
    quantity = Quantity(quantity_name)

    histogram = hist.Hist(
        hist.axis.Variable(quantity.bins, name=quantity_name, label=quantity.label),
        hist.axis.StrCategory(dataset_labels.keys(), name="label", label="label"),
        storage=hist.storage.Weight(),
    )

    return histogram


def get_legend_label(process_label: str) -> str:
    """Given a process label, return the corresponding legend label."""
    mapping = {
        "ewkz_17": "EWK Z(vv)",
        "ewkw_17": "EWK W(lv)",
        "qcdz_17": "QCD Z(vv)",
        "qcdw_17": "QCD W(lv)",
        "vbfh_17": "VBF H(inv)",
    }
    try:
        return mapping[process_label]
    except KeyError:
        raise RuntimeError(f"Cannot find legend entry for label: {process_label}")


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.pass_context
@click.argument("input_dir")
@click.option(
    "-n",
    "--normalize",
    is_flag=True,
    help="If specified, the weight normalization (per class) will be applied to the samples.",
)
def fill(ctx, input_dir: str, normalize: bool) -> None:
    """
    Fill feature histograms from the ROOT files located in the input directory.
    """
    all_datasets = load_datasets_bucoffea(input_dir)
    datasets = select_and_label_datasets(all_datasets, dataset_labels)
    summarize_datasets(datasets)

    features_to_plot = [
        "mjj",
        "detajj",
        "leadak4_eta",
        "trailak4_eta",
    ]

    sequence = build_sequence(
        datasets=copy.deepcopy(datasets),
        features=features_to_plot,
        weight_expression="weight_total*xs/sumw",
        shuffle=True,
        scale_features="none",
    )
    # Read the whole samples
    sequence.read_range = (0, 1)
    sequence.batch_size = int(1e5)
    sequence.batch_buffer_size = 10

    if normalize:
        normalize_classes(sequence)

    histograms = {}

    for ibatch in tqdm(range(len(sequence)), desc="Analyzing batches"):
        features, labels_onehot, weights = sequence[ibatch]
        labels = labels_onehot.argmax(axis=1)

        def get_string_labels(int_label: int):
            return list(dataset_labels.keys())[int_label]

        str_labels = np.array(list(map(get_string_labels, labels)))

        weights = weights.flatten()

        for iquantity, quantity_name in enumerate(features_to_plot):
            if quantity_name not in histograms:
                histograms[quantity_name] = make_histogram(quantity_name)

            histograms[quantity_name].fill(
                **{
                    quantity_name: features[:, iquantity],
                    "label": str_labels,
                    "weight": weights,
                }
            )

    # Save the histograms to cache
    outtag = os.path.basename(input_dir.rstrip("/"))
    if normalize:
        outdir = f"./output/{outtag}/normalized"
    else:
        outdir = f"./output/{outtag}"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cache = pjoin(outdir, "histograms.pkl")
    with open(cache, "wb+") as f:
        pickle.dump(histograms, f)


@cli.command()
@click.pass_context
@click.argument("cache_file")
def plot(ctx, cache_file: str) -> None:
    """
    Create a stack plot of histograms from the cache.
    """
    with open(cache_file, "rb") as f:
        histograms = pickle.load(f)

    outdir = os.path.dirname(cache_file)

    for quantity, histogram in tqdm(histograms.items(), desc="Plotting histograms"):
        fig, ax = plt.subplots()

        # Plot background as stack
        bkg_labels = [label for label in dataset_labels.keys() if label != "vbfh_17"]
        bkg_stack = [histogram[{"label": label}] for label in bkg_labels]
        hep.histplot(
            bkg_stack,
            stack=True,
            ax=ax,
            histtype="fill",
            label=list(map(get_legend_label, bkg_labels)),
        )

        # Plot the signal
        signal_proc = "vbfh_17"
        h_signal = histogram[{"label": signal_proc}]
        hep.histplot(h_signal, ax=ax, label=get_legend_label(signal_proc), color="k")

        ax.set_ylabel("Weighted Counts")

        ax.legend(title="Dataset")

        ax.set_yscale("log")

        outpath = pjoin(outdir, f"{quantity}.pdf")
        fig.savefig(outpath)
        plt.close(fig)


if __name__ == "__main__":
    cli()
