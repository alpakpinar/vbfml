#!/usr/bin/env python3

import os
from vbfml.training.plot import TrainingHistogramPlotter, history_plot
from vbfml.training.data import TrainingLoader
import click

from vbfml.training.analysis import TrainingAnalyzer


@click.group()
def cli():
    pass


@cli.command()
@click.argument("training_path")
def analyze(training_path):
    analyzer = TrainingAnalyzer(training_path)
    analyzer.analyze()
    analyzer.write_to_cache()


@cli.command()
@click.argument("training_path")
def plot(training_path):
    # Redo the analysis if cache does not exist
    analyzer = TrainingAnalyzer(training_path)
    if not analyzer.load_from_cache():
        analyzer.analyze()
        analyzer.write_to_cache()

    # Plot histograms
    output_directory = os.path.join(training_path, "plots")
    plotter = TrainingHistogramPlotter(analyzer.histograms, output_directory)
    plotter.plot()

    # Plot training history
    loader = TrainingLoader(training_path)
    history_plot(loader.get_history(), output_directory)


if __name__ == "__main__":
    cli()
