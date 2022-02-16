#!/usr/bin/env python3

import os

import click
import warnings
import pandas as pd

from vbfml.training.analysis import TrainingAnalyzer, ImageTrainingAnalyzer
from vbfml.training.accumulate import ImageAccumulator
from vbfml.training.data import TrainingLoader
from vbfml.training.plot import (
    ImageTrainingPlotter,
    TrainingHistogramPlotter,
    plot_history,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join


def get_model_arch(training_path):
    """
    Get the model architecture from a file called "model_identifier.txt".
    This will be used to determine which analyzer/plotter to call in these functions.

    Args:
        training_path ([type]): Path to the training directory, where the "model_identifier.txt" file is located.

    """
    filepath = pjoin(training_path, "model_identifier.txt")
    with open(filepath, "r") as f:
        arch = f.read().strip()
    return arch


@click.group()
def cli():
    pass


@cli.command()
@click.argument("training_path")
def analyze(training_path):
    arch = get_model_arch(training_path)

    analyzerInstances = {"conv": ImageTrainingAnalyzer, "dense": TrainingAnalyzer}
    analyzer = analyzerInstances[arch](training_path)
    analyzer.analyze()
    analyzer.write_to_cache()


@cli.command()
@click.argument("training_path")
@click.option("--force-analyze", default=False, is_flag=True)
def plot(training_path: str, force_analyze: bool = False):
    arch = get_model_arch(training_path)
    # Redo the analysis if cache does not exist
    analyzerInstances = {"conv": ImageTrainingAnalyzer, "dense": TrainingAnalyzer}
    plotterInstances = {
        "conv": ImageTrainingPlotter,
        "dense": TrainingHistogramPlotter,
    }
    analyzer = analyzerInstances[arch](training_path)
    if force_analyze or not analyzer.load_from_cache():
        analyzer.analyze()
        analyzer.write_to_cache()

    # Plot histograms
    output_directory = os.path.join(training_path, "plots")
    plotter_args = {
        "weights": analyzer.data["weights"],
        "predicted_scores": analyzer.data["predicted_scores"],
        "validation_scores": analyzer.data["validation_scores"],
        "histograms": analyzer.data["histograms"],
        "output_directory": output_directory,
    }
    if arch == "conv":
        plotter_args["grouped_image_data"] = analyzer.data["grouped_image_data"]
        plotter_args["sample_counts"] = analyzer.data["sample_counts_per_sequence"]
        plotter_args["label_encoding"] = analyzer.data["label_encoding"]

    plotter = plotterInstances[arch](**plotter_args)
    plotter.plot()

    if arch == "dense":
        plotter.plot_covariance(analyzer.data["covariance"])

    # Plot training history
    loader = TrainingLoader(training_path)
    plot_history(loader.get_history(), output_directory)


@cli.command()
@click.argument("training_path")
def accumulate(training_path: str):
    # This only makes sense if we're looking at image data
    arch = get_model_arch(training_path)
    if arch != "conv":
        raise RuntimeError(f"Cannot accumulate input data for model: {arch}")

    acc = ImageAccumulator(training_path)
    for groupby in ["truth_label", "predicted_label"]:
        acc.accumulate(groupby=groupby)
        acc.plot(groupby=groupby)


if __name__ == "__main__":
    cli()
