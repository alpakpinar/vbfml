#!/usr/bin/env python3

import os
import re
import copy
import hist
import click
import pickle
import warnings
import numpy as np
import pandas as pd
import mplhep as hep

from tqdm import tqdm
from matplotlib import pyplot as plt

from vbfml.training.input import build_sequence, load_datasets_bucoffea
from vbfml.training.util import (
    select_and_label_datasets,
    normalize_classes,
    save,
)
from vbfml.training.data import TrainingLoader
from vbfml.util import vbfml_path

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join

eta_bins = np.linspace(-5, 5, 50)
axis_bins = {
    "mjj": np.logspace(2, 4, 20),
    "detajj": np.linspace(1, 10, 30),
    "leadak4_eta": eta_bins,
    "trailak4_eta": eta_bins,
}


def get_directory_from_ctx(ctx) -> str:
    input_dir = ctx.obj["INPUT_DIR"]
    tag = os.path.basename(input_dir.rstrip("/"))
    return f"./output/{tag}/compare_distributions"


def make_histogram(quantity_name: str) -> hist.Hist:
    bins = axis_bins[quantity_name]
    histogram = hist.Hist(
        hist.axis.Variable(bins, name=quantity_name, label=quantity_name),
        hist.axis.IntCategory(range(10), name="label", label="label"),
        storage=hist.storage.Weight(),
    )
    return histogram


@click.group()
@click.option(
    "-i",
    "--input_dir",
    default=vbfml_path("root/2021-11-13_vbfhinv_treesForML"),
    required=False,
    help="Input directory containing the ROOT files for training and validation.",
)
@click.pass_context
def cli(ctx, input_dir):
    ctx.ensure_object(dict)
    ctx.obj["INPUT_DIR"] = input_dir


@cli.command()
@click.pass_context
def setup(ctx):
    """
    Set up sequences for training and validation.
    """
    input_dir = ctx.obj["INPUT_DIR"]
    all_datasets = load_datasets_bucoffea(input_dir)

    dataset_labels = {
        "ewk_17": "(EWK.*2017|VBF_HToInvisible_M125_withDipoleRecoil_pow_pythia8_2017)",
        "v_qcd_nlo_17": "(WJetsToLNu_Pt-\d+To.*|Z\dJetsToNuNu_M-50_LHEFilterPtZ-\d+To\d+)_MatchEWPDG20-amcatnloFXFX_2017",
    }
    datasets = select_and_label_datasets(all_datasets, dataset_labels)

    features = [
        "mjj",
        "detajj",
        "leadak4_eta",
        "trailak4_eta",
    ]

    weight_expression = "weight_total*xs/sumw"

    training_sequence = build_sequence(
        datasets=copy.deepcopy(datasets),
        features=features,
        weight_expression=weight_expression,
        shuffle=True,
        scale_features="none",
    )
    validation_sequence = build_sequence(
        datasets=copy.deepcopy(datasets),
        features=features,
        weight_expression=weight_expression,
        shuffle=True,
        scale_features="none",
    )

    normalize_classes(training_sequence)
    normalize_classes(validation_sequence)

    train_size = 0.8
    training_sequence.read_range = (0.0, train_size)
    training_sequence.batch_size = int(1e6)
    training_sequence.batch_buffer_size = 1

    validation_sequence.read_range = (train_size, 1.0)
    validation_sequence.batch_size = int(1e6)
    validation_sequence.batch_buffer_size = 1

    # Training and validation sequences
    # Clear buffer before saving to save space
    for seq in training_sequence, validation_sequence:
        seq.buffer.clear()

    outdir = get_directory_from_ctx(ctx)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save(training_sequence, pjoin(outdir, "training_sequence.pkl"))
    save(validation_sequence, pjoin(outdir, "validation_sequence.pkl"))


@cli.command()
@click.pass_context
def compare(ctx):
    """
    Compare training and validation sets.
    Save histograms of several quantities separately for
    training and validation datasets.
    """
    indir = get_directory_from_ctx(ctx)
    loader = TrainingLoader(indir)

    histograms = {}

    quantities = ["mjj", "detajj", "leadak4_eta", "trailak4_eta"]

    for sequence_type in ["training", "validation"]:
        sequence = loader.get_sequence(sequence_type)
        histograms[sequence_type] = {}
        for ibatch in tqdm(
            range(len(sequence)), desc=f"Analyzing {sequence_type} sequence"
        ):
            features, labels_onehot, weights = sequence[ibatch]
            labels = labels_onehot.argmax(axis=1)
            weights = weights.flatten()

            for idx, quantity in enumerate(quantities):
                if not quantity in histograms:
                    histograms[sequence_type][quantity] = make_histogram(quantity)

                histograms[sequence_type][quantity].fill(
                    **{
                        quantity: features[:, idx],
                        "label": labels,
                        "weight": weights,
                    }
                )

    # Write to cache
    cache = pjoin(indir, "histograms.pkl")
    with open(cache, "wb+") as f:
        pickle.dump(histograms, f)


@cli.command()
@click.pass_context
def plot(ctx):
    """
    Generates the histogram plots.
    """
    indir = get_directory_from_ctx(ctx)
    cache = pjoin(indir, "histograms.pkl")
    with open(cache, "rb") as f:
        histograms = pickle.load(f)

    for sequence_type, histo_dict in tqdm(
        histograms.items(), desc="Plotting histograms"
    ):
        for quantity, histogram in histo_dict.items():
            fig, ax = plt.subplots()

            label_to_proc = {
                0: "VBF 2017",
                1: "NLO QCD V 2017",
            }

            for label in [0, 1]:
                histogram_for_label = histogram[{"label": label}]
                edges = histogram_for_label.axes[0].edges
                values = histogram_for_label.values()

                hep.histplot(
                    values,
                    edges,
                    label=label_to_proc[label],
                    ax=ax,
                )

            ax.set_xlabel(quantity)
            ax.set_ylabel("Counts")
            ax.set_yscale("log")
            ax.legend()

            if quantity == "mjj":
                ax.set_xscale("log")

            ax.text(
                0,
                1,
                sequence_type,
                fontsize=14,
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )

            outpath = pjoin(indir, f"{sequence_type}_{quantity}.pdf")
            fig.savefig(outpath)


if __name__ == "__main__":
    cli()
