#!/usr/bin/env python3
import copy
import os
import re
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

import click
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD

from vbfml.models import sequential_dense_model, sequential_convolutional_model
from vbfml.training.data import TrainingLoader
from vbfml.training.input import build_sequence, load_datasets_bucoffea
from vbfml.training.util import (
    append_history,
    normalize_classes,
    save,
    select_and_label_datasets,
    scale_datasets,
    summarize_datasets,
    summarize_labels,
    PrintingCallback,
)
from vbfml.util import (
    ModelConfiguration,
    ModelFactory,
    DatasetAndLabelConfiguration,
    vbfml_path,
    write_repo_version,
    write_model_info,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pjoin = os.path.join


@click.group()
@click.option(
    "-d",
    "--training-directory",
    required=True,
    help="Name of the output directory to save data and the model.",
)
@click.pass_context
def cli(ctx, training_directory):
    ctx.ensure_object(dict)
    if not os.path.exists(training_directory):
        os.makedirs(training_directory)
    ctx.obj["TRAINING_DIRECTORY"] = training_directory


@cli.command()
@click.pass_context
@click.option(
    "--learning-rate", default=1e-3, required=False, help="Learning rate for training."
)
@click.option(
    "--input-dir",
    default=vbfml_path("root/2021-11-13_vbfhinv_treesForML"),
    required=False,
    help="Input directory containing the ROOT files for training and validation.",
)
@click.option(
    "-m",
    "--model-config",
    required=True,
    help="Path to the .yml file that has the model configuration parameters.",
)
@click.option(
    "--no-plot-model",
    is_flag=True,
    help="Flag to disable the plot_model call via Keras.",
)
def setup(
    ctx,
    learning_rate: float,
    input_dir: str,
    model_config: str,
    no_plot_model: bool = False,
):
    """
    Creates a new working area. Prerequisite for later training.

    By default, if the model is a dense neural network (DNN), this function
    will setup a PyTorch model, and if it is a convolutional neural network (CNN),
    it will setup a Keras model.
    """

    all_datasets = load_datasets_bucoffea(input_dir)

    # Get datasets and corresponding labels from datasets.yml
    datasets_path = vbfml_path("config/datasets/datasets.yml")
    dataset_config = DatasetAndLabelConfiguration(datasets_path)

    dataset_labels = dataset_config.get_dataset_labels()

    datasets = select_and_label_datasets(all_datasets, dataset_labels)
    scale_datasets(datasets, dataset_config)
    summarize_datasets(datasets)

    # Object containing data for different models
    # (set of features, dropout rate etc.)
    # Loaded from the YML configuration file
    mconfig = ModelConfiguration(model_config)

    features = mconfig.get("features")

    training_params = mconfig.get("training_parameters")
    validation_params = mconfig.get("validation_parameters")

    training_sequence = build_sequence(
        datasets=copy.deepcopy(datasets),
        features=features,
        weight_expression=mconfig.get("weight_expression"),
        shuffle=training_params["shuffle"],
        scale_features=training_params["scale_features"],
    )
    validation_sequence = build_sequence(
        datasets=copy.deepcopy(datasets),
        features=features,
        weight_expression=mconfig.get("weight_expression"),
        shuffle=validation_params["shuffle"],
        scale_features=validation_params["scale_features"],
    )

    summarize_labels(training_sequence, dataset_config)

    normalize_classes(training_sequence, target_integral=1e6)
    normalize_classes(validation_sequence, target_integral=1e6)

    # Training sequence
    train_size = training_params["train_size"]

    training_sequence.read_range = (0.0, train_size)
    training_sequence.batch_size = training_params["batch_size"]
    training_sequence.batch_buffer_size = training_params["batch_buffer_size"]
    training_sequence[0]

    # Validation sequence
    validation_sequence.read_range = (train_size, 1.0)
    validation_sequence._feature_scaler = copy.deepcopy(
        training_sequence._feature_scaler
    )
    validation_sequence.batch_size = validation_params["batch_size"]
    validation_sequence.batch_buffer_size = validation_params["batch_buffer_size"]

    # Build model
    # We're assuming that if architecture = "dense", this is a PyTorch model
    use_pytorch = mconfig.get("architecture") == "dense"

    model = ModelFactory.build(mconfig)
    if use_pytorch:
        print("\nPyTorch DNN model:")
        print(f"\n{model}\n")

    # Keras-related model operations
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            weighted_metrics=["categorical_accuracy"],
        )
        model.summary()

    training_directory = ctx.obj["TRAINING_DIRECTORY"]

    def prepend_path(fname):
        return os.path.join(training_directory, fname)

    # Save the model for later training
    if use_pytorch:
        torch.save(model, prepend_path("model.pt"))
    else:
        model.save(
            os.path.join(training_directory, "models/initial"), include_optimizer=True
        )
        model.save(
            os.path.join(training_directory, "models/latest"), include_optimizer=True
        )

    # Feature scaling object for future evaluation
    save(training_sequence._feature_scaler, prepend_path("feature_scaler.pkl"))

    # List of features
    save(
        features,
        prepend_path(
            "features.pkl",
        ),
    )

    # Training and validation sequences
    # Clear buffer before saving to save space
    for seq in training_sequence, validation_sequence:
        seq.buffer.clear()
    save(training_sequence, prepend_path("training_sequence.pkl"))
    save(validation_sequence, prepend_path("validation_sequence.pkl"))

    # Save model type identifier for later uses
    with open(prepend_path("model_identifier.txt"), "w+") as f:
        f.write(mconfig.get("architecture"))

    # Save repo version information to version.txt
    write_repo_version(os.path.join(training_directory, "version.txt"))

    # Save the arch parameters for this model as a table
    write_model_info(mconfig, os.path.join(training_directory, "arch.txt"))

    # Save a plot of the model architecture (only supported for Keras models)
    if not no_plot_model and not use_pytorch:
        from keras.utils.vis_utils import plot_model

        plot_dir = os.path.join(training_directory, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_file = os.path.join(plot_dir, "model.png")
        plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)


@cli.command()
@click.pass_context
@click.option(
    "-n",
    "--num-epochs",
    type=int,
    default=10,
    help="Number of iterations through the whole training set.",
)
@click.option(
    "--learning-rate", type=float, default=None, help="Set new learning rate."
)
@click.option(
    "--no-verbose-output",
    is_flag=True,
    help="""
    Do not use the regular Keras output, instead, 
    use the customized printing callback.
    Mainly used for HTCondor submissions.
    """,
)
def train(
    ctx,
    num_epochs: int,
    learning_rate: float,
    no_verbose_output: bool,
):
    """
    Train in a previously created working area.
    """
    training_directory = ctx.obj["TRAINING_DIRECTORY"]
    filepath = pjoin(training_directory, "model_identifier.txt")
    with open(filepath, "r") as f:
        arch = f.read().strip()
    loader = TrainingLoader(training_directory, arch=arch)

    model = loader.get_model("latest")

    if learning_rate:
        assert learning_rate > 0, "Learning rate should be positive."
        if arch != "dense":
            K.set_value(model.optimizer.learning_rate, learning_rate)

    training_sequence = loader.get_sequence("training")
    validation_sequence = loader.get_sequence("validation")
    # assert training_sequence._feature_scaler
    # assert validation_sequence._feature_scaler

    validation_freq = 1  # Frequency of validation
    if arch == "dense":
        device = torch.device("cpu")
        # Binary cross entropy loss
        # Do not apply reduction, so that we can implement
        # manual weighted reduction later on
        history = defaultdict(list)
        criterion = nn.BCELoss(reduction="none")
        # criterion =  nn.CrossEntropyLoss(reduction='none')
        # criterion =  nn.MSELoss(reduction='none')
        optm = SGD(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optm,
            factor=0.1,
            patience=5,
            threshold=1e-2,
            verbose=True,
            cooldown=2,
            min_lr=1e-5,
        )

        def train_pytorch(
            model,
            features: torch.Tensor,
            true_labels: torch.Tensor,
            weights: torch.Tensor,
            optimizer,
            loss_criterion,
        ):
            """
            One iteration of training on a batch of data.
            """
            model.zero_grad()
            loss, predicted_labels = weighted_loss(
                model, features, true_labels, weights, loss_criterion
            )
            loss.backward()
            optimizer.step()
            return loss, predicted_labels

        def weighted_loss(model, features, true_labels, weights, loss_criterion):
            """
            Compute the weighted loss for a batch of data.
            """
            predicted_labels = model(features)
            # print(predicted_labels)
            raw_loss = loss_criterion(predicted_labels, true_labels)
            loss = torch.sum(weights * raw_loss) / torch.sum(weights)
            return loss, predicted_labels

        for epoch in range(num_epochs):
            training_loss = 0
            correct = 0

            # Put model into training mode
            model.train(True)

            # Loop over training batches and train
            for batch in tqdm(training_sequence, desc="Running training"):
                x_train, y_train, w_train = batch
                loss, predictions = train_pytorch(
                    model,
                    torch.Tensor(x_train).to(device),
                    torch.Tensor(y_train).to(device),
                    torch.Tensor(w_train).to(device),
                    optm,
                    criterion,
                )
                training_loss += loss.item()

            # Put model into inference mode
            model.train(False)

            # Calculate validation loss
            validation_loss = 0
            for batch in validation_sequence:
                x, y, w = batch
                loss, _ = weighted_loss(
                    model,
                    torch.Tensor(x).to(device),
                    torch.Tensor(y).to(device),
                    torch.Tensor(w).to(device),
                    criterion,
                )
                validation_loss += loss.item()

            # Normalize losses by number of batches
            training_loss /= len(training_sequence)
            validation_loss /= len(validation_sequence)

            # Pass validation loss to learning rate scheduler
            scheduler.step(validation_loss)

            print(
                f"Epoch {epoch+1} Loss : {training_loss:.4f} Validation loss : {validation_loss:.4f}"
            )

            history["x_loss"].append(epoch)
            history["x_val_loss"].append(epoch)
            history["y_loss"].append(training_loss)
            history["y_val_loss"].append(validation_loss)

        def prepend_path(fname):
            return os.path.join(training_directory, fname)

        # save The trained model
        torch.save(model, prepend_path("model.pt"))
    # Keras-based training
    else:
        fit_args = {
            "x": training_sequence,
            "epochs": num_epochs,
            "max_queue_size": 0,
            "shuffle": False,
            "validation_data": validation_sequence,
            "validation_freq": validation_freq,
        }

        # Use the less-verbose printing
        if no_verbose_output:
            fit_args["  "] = 0
            fit_args["callbacks"] = [PrintingCallback()]

        # Run the training
        model.fit(**fit_args)
        try:
            history = loader.get_history()
        except:
            history = {}
        history = append_history(
            history, model.history.history, validation_frequence=validation_freq
        )

        model.save(
            os.path.join(training_directory, "models/latest"), include_optimizer=True
        )

    def prepend_path(fname):
        return os.path.join(training_directory, fname)

    save(history, prepend_path("history.pkl"))


if __name__ == "__main__":
    cli()
