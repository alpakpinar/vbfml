#!/usr/bin/env python3

import os
import sys
import re
import warnings
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
from glob import glob
from typing import Dict, List, Optional
from pprint import pprint

from vbfml.training.data import TrainingLoader
from vbfml.training.util import load
from vbfml.helpers.compare import HistorySet
from vbfml.util import vbfml_path

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join


def main():
    training_directory = sys.argv[1]

    h = HistorySet(training_directory)
    h.accumulate_histories()

    quantities = [
        "loss",
        "val_loss",
        "categorical_accuracy",
        "val_categorical_accuracy",
    ]
    for quantity in tqdm(quantities, desc="Plotting training metrics"):
        h.plot(quantity)


if __name__ == "__main__":
    main()
