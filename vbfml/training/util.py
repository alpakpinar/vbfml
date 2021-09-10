import pickle
from datetime import datetime

import boost_histogram as bh
import tensorflow as tf
import uproot
from tqdm import tqdm


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


def get_weight_integral_by_label(sequence) -> "dict[str:float]":
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


def normalize_classes(sequence: "MultiDatasetSequence") -> None:
    """Changes weights so that all classes have same integral."""
    label_to_weight_dict = get_weight_integral_by_label(sequence)
    for dataset_obj in sequence.datasets.values():
        weight = label_to_weight_dict[dataset_obj.label]
        dataset_obj.weight = dataset_obj.weight / weight


def generate_callbacks_for_profiling():
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
