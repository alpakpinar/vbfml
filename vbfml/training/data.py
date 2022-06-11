import os

from vbfml.util import get_model_arch
from vbfml.training.util import load


class TrainingLoader:
    """
    Class to load data from a training area.

    When loading the model, this class keeps track of whether to load
    a PyTorch model or Keras model, based on the arch given in the constructor.
    """

    def __init__(self, training_directory):
        self._directory = os.path.abspath(training_directory)
        self._arch = get_model_arch(training_directory)

    def _fpath(self, fname):
        return os.path.join(self._directory, fname)

    def get_model(self, tag: str = "latest"):
        """Loads and returns the neural network model."""
        if self._arch == "conv":
            import tensorflow as tf

            return tf.keras.models.load_model(
                os.path.join(self._directory, f"models/{tag}")
            )
        elif self._arch == "dense":
            import torch

            return torch.load(os.path.join(self._directory, "model.pt"))

        raise RuntimeError(f"Cannot load model of given architecture: {self._arch}")

    def get_sequence(self, sequence_type="training"):
        return load(self._fpath(f"{sequence_type}_sequence.pkl"))

    def get_features(self):
        return load(self._fpath("features.pkl"))

    def get_history(self):
        return load(self._fpath("history.pkl"))

    def get_feature_scaler(self):
        return load(self._fpath("feature_scaler.pkl"))
