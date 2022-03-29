import os
import re
import copy
import pickle
import numpy as np

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

import boost_histogram as bh
import tensorflow as tf
import uproot
from tabulate import tabulate
from tqdm import tqdm

from vbfml.input.sequences import DatasetInfo, MultiDatasetSequence
from vbfml.util import ModelConfiguration, DatasetAndLabelConfiguration, vbfml_path

pjoin = os.path.join


def load(fpath: str) -> object:
    with open(fpath, "rb") as f:
        return pickle.load(f)


def save(data: object, fpath: str) -> None:
    with open(fpath, "wb") as f:
        return pickle.dump(data, f)


def get_n_events(filename: str, treename: str) -> int:
    """Return number of events in a TTree in a given file"""
    try:
        return int(uproot.open(filename)[treename].num_entries)
    except uproot.exceptions.KeyInFileError:
        return 0


def get_total_n_events(filelist: List[str], treename: str) -> int:
    """Return the total number of events in a list of files."""
    n_events = 0
    for file in filelist:
        n_events += get_n_events(file, treename)
    return n_events


def get_weight_integral_by_label(sequence: MultiDatasetSequence) -> Dict[str, float]:
    """
    Integrate the weights of all samples, accumulate by their label.
    """
    # Save settings so we can restore later
    backup = {}
    for key in ["batch_size", "batch_buffer_size"]:
        backup[key] = getattr(sequence, key)

    # For integration, batches can be large
    sequence.batch_size = int(1e6)
    sequence.batch_buffer_size = 10

    # Histogram to handle storage
    string_labels = sequence.dataset_labels()
    total_weight = bh.Histogram(
        bh.axis.IntCategory(list(range(len(string_labels))), growth=False),
        storage=bh.storage.Weight(),
    )

    # Batch loop
    for i in tqdm(range(len(sequence)), desc="Determining weight integrals"):
        _, labels, weight = sequence[i]
        total_weight.fill(labels.argmax(axis=1), weight=weight)

    # Integrals are separated by truth label
    integrals = {}
    for i, label in enumerate(string_labels):
        h = total_weight[{0: i}]
        integrals[label] = h.value

    # Restore original settings
    for key, value in backup.items():
        setattr(sequence, str(key), value)

    return integrals


def normalize_classes(sequence: MultiDatasetSequence) -> None:
    """Changes data set weights in-place so that all classes have same integral."""
    label_to_weight_dict = get_weight_integral_by_label(sequence)
    for dataset_obj in sequence.datasets.values():
        weight = label_to_weight_dict[dataset_obj.label]
        dataset_obj.weight = dataset_obj.weight / weight


def generate_callbacks_for_profiling() -> None:
    """
    Callbacks for profiling of keras training.

    See https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
    """

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs, histogram_freq=1, profile_batch="1,1000"
    )
    callbacks = [tboard_callback]

    return callbacks


def summarize_datasets(datasets: List[DatasetInfo]) -> None:
    """
    Prints a neat summary of a group of datasets to the terminal.
    """
    table = []
    total_by_label = defaultdict(int)
    for dataset in datasets:
        table.append((dataset.label, dataset.name, dataset.n_events))
        total_by_label[dataset.label] += dataset.n_events

    for key, value in total_by_label.items():
        table.append((key, "--- TOTAL ---", value))

    print(
        tabulate(
            sorted(table),
            headers=["Class label", "Physics data set name", "Number of events"],
        )
    )


def scale_datasets(
    datasets: List[DatasetInfo], dataset_config: DatasetAndLabelConfiguration
) -> None:
    """
    Based on the dataset configuration given, scale events for each label in place.
    """
    labels = dataset_config.get_dataset_labels()
    scales = dataset_config.get_dataset_scales()
    for label, scale in scales.items():
        for dataset_info in datasets:
            if re.match(labels[label], dataset_info.name):
                dataset_info.n_events = int(np.floor(scale * dataset_info.n_events))


def select_and_label_datasets(
    all_datasets: List[DatasetInfo], labels: Dict[str, str]
) -> List[DatasetInfo]:
    """
    Slim down a list of datasets and apply labeling.

    Labels is a dictionary that maps a new target label name to a regular
    expression matching data set names. Only data sets matching one of the
    regular expressions are returned. The labels of these data sets are
    set to the corresponding key from the labels dict.
    """
    selected_datasets = []
    for label, regex in labels.items():
        matching_datasets = [
            deepcopy(dataset)
            for dataset in all_datasets
            if re.match(regex, dataset.name)
        ]
        for dataset in matching_datasets:
            dataset.label = label
        selected_datasets.extend(matching_datasets)
    return selected_datasets


def append_history(
    history1: Dict[str, List[float]],
    history2: Dict[str, List[float]],
    validation_frequence: int = 1,
) -> Dict[str, List[float]]:
    """
    Append keras training histories.

    The second history is appended to the first one, and the combined history is returned.
    """
    new_history = {}
    for key, value_list in history2.items():
        n_entries = len(value_list)

        x_freq = 1
        x_offset = 0

        if "val" in key:
            x_freq = validation_frequence
            x_offset = 1

        if key in history1:
            original_x = history1[f"x_{key}"]
            new_x = original_x + [
                original_x[-1] + (ix + x_offset) * x_freq for ix in range(n_entries)
            ]
            new_y = history1[key] + value_list
        else:
            new_x = [(ix + x_offset) * x_freq for ix in range(n_entries)]
            new_y = value_list

        new_history[f"x_{key}"] = new_x
        new_history[f"y_{key}"] = new_y
    return new_history


def do_setup(
    output_directory: str,
    input_dir: str,
    model_config: str,
):
    """
    Creates a new working area with training and valdation sequences.
    Prerequisite for later training.
    """
    from vbfml.training.input import build_sequence, load_datasets_bucoffea

    all_datasets = load_datasets_bucoffea(directory=input_dir)

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
    normalize_classes(training_sequence)
    normalize_classes(validation_sequence)

    # Training sequence
    train_size = training_params["train_size"]

    training_sequence.read_range = (0.0, train_size)
    training_sequence.batch_size = training_params["batch_size"]
    training_sequence.batch_buffer_size = training_params["batch_buffer_size"]
    training_sequence[0]

    # Validation sequence
    validation_sequence.read_range = (train_size, 1.0)
    validation_sequence.scale_features = True
    validation_sequence._feature_scaler = copy.deepcopy(
        training_sequence._feature_scaler
    )
    validation_sequence.batch_size = validation_params["batch_size"]
    validation_sequence.batch_buffer_size = validation_params["batch_buffer_size"]

    try:
        os.makedirs(output_directory)
    except FileExistsError:
        pass

    def prepend_path(fname):
        return pjoin(output_directory, fname)

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


@dataclass
class PrintingCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to control training output.
    Especially useful if we don't want to log the verbose output of Keras to HTCondor logs.
    To use this callback during model training, use:

    >>> model.fit(
        ...,
        verbose=0,
        callbacks=[PrintingCallback()],
        ...
    )
    """

    SHOW_NUMBER: int = 1

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.SHOW_NUMBER == 0 or epoch == 0:
            print(
                f'Epoch: {epoch:5} Loss: {logs["loss"]:.4e}, accuracy: {logs["categorical_accuracy"]:.4f}, val_loss: {logs["val_loss"]:.4e}, val_accuracy: {logs["val_categorical_accuracy"]:.4f}'
            )
