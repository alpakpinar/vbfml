#!/usr/bin/env python3

import os

import click
import warnings
import pandas as pd

from vbfml.training.analysis import TrainingAnalyzer, ImageTrainingAnalyzer
from vbfml.training.data import TrainingLoader
from vbfml.training.plot import (
    ImageTrainingPlotter,
    TrainingHistogramPlotter,
    plot_history,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("training_path")
@click.option("--type", default="image", required=False)
def analyze(training_path, type):
    analyzerInstances = {"image": ImageTrainingAnalyzer, "dense": TrainingAnalyzer}
    analyzer = analyzerInstances[type](training_path)
    analyzer.analyze()
    analyzer.write_to_cache()


@cli.command()
@click.argument("training_path")
@click.option("--type", default="image", required=False)
@click.option("--force-analyze", default=False, is_flag=True)
def plot(training_path: str, type: str, force_analyze: bool = False):
    # Redo the analysis if cache does not exist
    analyzerInstances = {"image": ImageTrainingAnalyzer, "dense": TrainingAnalyzer}
    plotterInstances = {
        "image": ImageTrainingPlotter,
        "dense": TrainingHistogramPlotter,
    }
    analyzer = analyzerInstances[type](training_path)
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
    if type == "image":
        plotter_args["features"] = analyzer.data["features"]

    plotter = plotterInstances[type](**plotter_args)
    plotter.plot()

    if type == "dense":
        plotter.plot_covariance(analyzer.data["covariance"])

    # Plot training history
    loader = TrainingLoader(training_path)
    plot_history(loader.get_history(), output_directory)


if __name__ == "__main__":
    cli()
