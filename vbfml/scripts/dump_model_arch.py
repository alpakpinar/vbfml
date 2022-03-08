#!/usr/bin/env python

import os
import sys
from vbfml.training.data import TrainingLoader
from keras.utils.vis_utils import plot_model


def dump_model_arch(training_directory):
    """Produces a nice plot of the model architecture."""
    loader = TrainingLoader(training_directory)
    model = loader.get_model()

    model.summary()

    # Produce the plot
    plot_dir = os.path.join(training_directory, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_file = os.path.join(plot_dir, "model.png")
    plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide the path to the training directory!"
    training_directory = sys.argv[1]
    dump_model_arch(training_directory)
