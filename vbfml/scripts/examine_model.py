#!/usr/bin/env python

import os
import click
import warnings
import numpy as np
import pandas as pd

from vbfml.training.data import TrainingLoader

pjoin = os.path.join


@click.group()
def cli():
    pass


@cli.command()
@click.argument("training_path")
@click.option(
    "--layer",
    default=-1,
    type=int,
    required=False,
    help="The layer index to look at for weights. Default is the last layer.",
)
def weights(training_path: str, layer: int):
    """
    Retrieve and display model weights.
    """
    loader = TrainingLoader(training_path)
    model = loader.get_model()

    weights = model.get_weights()[layer]
    print(weights)


if __name__ == "__main__":
    cli()
