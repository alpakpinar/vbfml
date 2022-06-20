#!/usr/bin/env python3

import os
import click
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.colors

from tqdm import tqdm
from matplotlib import pyplot as plt

from vbfml.training.data import TrainingLoader

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.pass_context
@click.argument("input_dir")
@click.option(
    "-n",
    "--num-batches",
    default=3,
    required=False,
    help="Number of batches to do the averaging for.",
)
@click.option(
    "-b",
    "--batch-size",
    default=int(1e4),
    required=False,
    help="Batch size for each averaging operation.",
)
def compute(ctx, input_dir: str, num_batches: int, batch_size: int):
    """
    For the training and validation sequences, computes and saves the
    average images (by energy) for several batches.
    """
    loader = TrainingLoader(input_dir)

    mean_images = {}
    class_names = ["ewk_17", "v_qcd_nlo_17"]

    for sequence_type in ["training", "validation"]:
        sequence = loader.get_sequence(sequence_type)

        sequence.batch_size = batch_size
        sequence.batch_buffer_size = num_batches

        mean_images[sequence_type] = {k: [] for k in class_names}

        for ibatch in tqdm(
            range(num_batches), desc=f"Averaging {sequence_type} batches"
        ):
            features, labels_onehot, weights = sequence[ibatch]

            labels = labels_onehot.argmax(axis=1)
            weights = weights.flatten()

            # Compute average per class
            for ilabel, label in enumerate(class_names):
                images = features[labels == ilabel]
                mean = np.average(images, axis=0, weights=weights[labels == ilabel])

                mean_images[sequence_type][label].append(mean)

    # When done, save everything into cache
    outtag = os.path.basename(input_dir.rstrip("/"))
    cachedir = f"./output/{outtag}/averaged_images"
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    cache = pjoin(cachedir, "averaged_images.pkl")
    with open(cache, "wb+") as f:
        pickle.dump(mean_images, f)


@cli.command()
@click.pass_context
@click.argument("input_file")
def plot(ctx, input_file: str):
    """
    Read the averaged images from cache and plot them.
    """
    with open(input_file, "rb") as f:
        averaged_images = pickle.load(f)

    outdir = os.path.dirname(input_file)

    for sequence_type, imagedict in averaged_images.items():
        for proc_type, imagelist in tqdm(
            imagedict.items(), desc=f"Plotting {sequence_type} images"
        ):
            for idx, image in enumerate(imagelist):
                fig, ax = plt.subplots()

                # Pre-processing
                imshape = (40, 20)
                image = image.reshape(imshape)

                eta_bins = np.linspace(-5, 5, imshape[0])
                phi_bins = np.linspace(-np.pi, np.pi, imshape[1])

                cmap = ax.pcolormesh(
                    eta_bins,
                    phi_bins,
                    image.T,
                    norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1e-2),
                )

                cb = fig.colorbar(cmap)
                cb.set_label("Averaged Weighted Energy per Pixel (GeV)")

                ax.text(
                    0,
                    1,
                    proc_type,
                    fontsize=13,
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                )

                ax.text(
                    1,
                    1,
                    sequence_type,
                    fontsize=13,
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                )

                ax.set_xlabel(r"PF Candidate $\eta$")
                ax.set_ylabel(r"PF Candidate $\phi$")

                outpath = pjoin(outdir, f"{sequence_type}_{idx}_{proc_type}.pdf")
                fig.savefig(outpath)
                plt.close(fig)


if __name__ == "__main__":
    cli()
