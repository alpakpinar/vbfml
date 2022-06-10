#!/usr/bin/env python3

import os
import click
import warnings
import pickle
import numpy as np
import pandas as pd

from tabulate import tabulate
from tqdm import tqdm
from collections import Counter

from vbfml.training.data import TrainingLoader

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join


@click.group()
@click.option(
    "-d",
    "--training-directory",
    required=True,
    help="Path to the training directory.",
)
@click.pass_context
def cli(ctx, training_directory):
    ctx.ensure_object(dict)
    ctx.obj["DIR"] = training_directory


@cli.command()
@click.pass_context
@click.option(
    "-s",
    "--sequence-type",
    default="training",
    required=False,
    help="Sequence type to examine.",
)
def examine(ctx, sequence_type: str):
    """
    For the given training directory, examine the sequence type.
    """
    training_directory = ctx.obj["DIR"]
    loader = TrainingLoader(training_directory)

    sequence = loader.get_sequence(sequence_type)

    dataset_fractions = []
    for dataset, frac in sequence.fractions.items():
        dataset_fractions.append([dataset, frac])

    print(tabulate(dataset_fractions, headers=["Dataset name", "Event fraction"]))
    sequence.batch_size = int(1e5)
    sequence.batch_buffer_size = 1
    sequence[0]

    buffer = sequence.buffer
    df = buffer.get_batch_df(0)

    neg_weights_mask = df["weight"] < 0
    for i_class in [0, 1]:
        mask = df["label"] == i_class
        print(len(df["weight"][mask]))
        print(len(df["weight"][neg_weights_mask & mask]))


@cli.command()
@click.pass_context
def accumulate(ctx):
    """
    For the training and validation sequences, accumulate the images
    and produce an averaged-out image per class.
    """
    training_directory = ctx.obj["DIR"]
    loader = TrainingLoader(training_directory)
    mean_images = {}

    sequence_types = ["training", "validation"]
    for sequence_type in sequence_types:
        sequence = loader.get_sequence(sequence_type)

        sequence.batch_size = int(1e3)
        sequence.batch_buffer_size = 10

        avg_images = {
            "ewk_17": [],
            "v_qcd_nlo_17": [],
        }

        for ibatch in tqdm(
            range(len(sequence)), desc=f"Accumulating {sequence_type} sequence"
        ):
            features, labels_onehot, weights = sequence[ibatch]
            labels = labels_onehot.argmax(axis=1)
            weights = weights.flatten()

            # EWK V images
            images_ewk = features[labels == 0]
            mean_ewk = np.average(images_ewk, axis=0, weights=weights[labels == 0])
            # QCD V images
            images_qcd = features[labels == 1]
            mean_qcd = np.average(images_qcd, axis=0, weights=weights[labels == 1])

            avg_images["ewk_17"].append(mean_ewk)
            avg_images["v_qcd_nlo_17"].append(mean_qcd)

        # Finally average out the full sets
        mean_images[sequence_type] = {}
        for key, images in avg_images.items():
            temp = np.average(np.array(images), axis=0)

            mean_images[sequence_type][key] = temp.reshape(40, 20)

    outdir = pjoin(training_directory, "accumulated_images")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    pkl_output = pjoin(outdir, "images.pkl")
    with open(pkl_output, "wb+") as f:
        pickle.dump(mean_images, f)


if __name__ == "__main__":
    cli()
