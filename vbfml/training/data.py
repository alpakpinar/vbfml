import os

import tensorflow as tf

from vbfml.training.util import load


class TrainingLoader:
    def __init__(self, training_directory):
        self._directory = os.path.abspath(training_directory)

    def _fpath(self, fname):
        return os.path.join(self._directory, fname)

    def get_model(self):
        return tf.keras.models.load_model(self._directory)

    def get_sequence(self, sequence_type="training"):
        return load(self._fpath(f"{sequence_type}_sequence.pkl"))

    def get_features(self):
        return load(self._fpath("features.pkl"))

    def get_history(self):
        return load(self._fpath("history.pkl"))

    def get_feature_scaler(self):
        return load(self._fpath("feature_scaler.pkl"))
