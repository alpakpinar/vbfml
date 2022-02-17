#!/usr/bin/env python3

import os
import warnings
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Dict

from vbfml.training.data import TrainingLoader
from vbfml.util import vbfml_path

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join



def compare(training_paths: Dict[str, str]) -> None:
    '''Retrieves and plots a comparison of loss functions for different trainings.'''
    losses = {}
    metrics = [
        ("Training", "x_loss", "y_loss"),
        ("Validation", "x_val_loss", "y_val_loss"),
    ]

    markers = {
        'Training' : 'o',
        'Validation' : '*',
    }
    
    def shift_by_one(xlist):
        return [x + 1 for x in xlist]

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12,8))

    for path, training_label in tqdm(training_paths.items()):
        loader = TrainingLoader(path)
        history = loader.get_history()

        for metric in metrics:
            label, x_label, y_label = metric
            x, y = history[x_label], history[y_label]

            if label == "Training":
                x = shift_by_one(x)
                ax1.plot(x, y, label=f'{training_label}', marker=markers[label])
            else:
                ax2.plot(x, y, label=f'{training_label}', marker=markers[label])
    
    for ax in (ax1, ax2):
        ax.grid(True)
        ax.legend()
        ax.set_yscale("log")
        
    ax1.set_ylabel('Training Loss', fontsize=14)
    ax2.set_ylabel('Validation Loss', fontsize=14)
    
    ax2.set_xlabel("Training Time (a.u.)", fontsize=14)

    outdir = './output/training_comparisons'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, f"loss_comparison.pdf")
    fig.savefig(outpath)
    plt.close(fig)

def main():
    training_paths = {
        vbfml_path('scripts/output/model_2022-02-17_trainsize_0_8') : 'Dropout = 0.0',
        vbfml_path('scripts/output/model_2022-02-17_dropout_0_2') : 'Dropout = 0.2',
        vbfml_path('scripts/output/model_2022-02-17_dropout_0_4') : 'Dropout = 0.4',
    }

    compare(training_paths)

if __name__ == "__main__":
    main()

    