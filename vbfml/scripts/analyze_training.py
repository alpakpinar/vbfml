#!/usr/bin/env python3

import os

import click
import warnings
import pandas as pd

from typing import List

from vbfml.training.analysis import TrainingAnalyzer, ImageTrainingAnalyzer
from vbfml.training.accumulate import (
    ImageAccumulatorFromAnalyzerCache,
)
from vbfml.training.data import TrainingLoader
from vbfml.training.plot import (
    ImageTrainingPlotter,
    TrainingHistogramPlotter,
    plot_history,
)

from vbfml.util import get_model_arch

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join


@click.group()
def cli():
    pass


@cli.command()
@click.argument("training_path")
@click.option("-s", "--sequence-types", multiple=True, default=["validation"])
def analyze(training_path: str, sequence_types: List[str]):
    """
    Given a training area, analyze the trained model and fill feature and score histograms.
    """
    arch = get_model_arch(training_path)
    analyzerInstances = {"conv": ImageTrainingAnalyzer, "dense": TrainingAnalyzer}
    analyzer = analyzerInstances[arch](training_path)
    analyzer.analyze()
    analyzer.write_to_cache()


@cli.command()
@click.argument("training_path")
@click.option("--force-analyze", default=False, is_flag=True)
def plot(training_path: str, force_analyze: bool = False):
    """
    Given a training area with the analyzer cache (output of the analyze function),
    plot the resulting histograms.
    """
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
        "truth_scores": analyzer.data["truth_scores"],
        "histograms": analyzer.data["histograms"],
        "output_directory": output_directory,
    }
    if arch == "conv":
        plotter_args["sample_counts"] = analyzer.data["sample_counts_per_sequence"]
        plotter_args["label_encoding"] = analyzer.data["label_encoding"]

    # Create the plotter instance and plot the histograms
    plotter = plotterInstances[arch](**plotter_args)
    plotter.plot()

    # Covariance plots for DNNs
    if arch == "dense":
        plotter.plot_covariance(analyzer.data["covariance"])

    # Plot training history
    loader = TrainingLoader(training_path)
    plot_history(loader.get_history(), output_directory)


if __name__ == "__main__":
    cli()
