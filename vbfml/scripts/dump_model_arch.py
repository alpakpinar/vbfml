#!/usr/bin/env python

import sys
from vbfml.training.data import TrainingLoader

def dump_model_arch(training_directory):
    loader = TrainingLoader(training_directory)
    model = loader.get_model()

    model.summary()

if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please provide the path to the training directory!"
    training_directory = sys.argv[1]
    dump_model_arch(training_directory)