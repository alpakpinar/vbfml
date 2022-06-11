import os


from vbfml.training.util import load


class TrainingLoader:
    """
    Class to load data from a training area.

    When loading the model, this class keeps track of whether to load
    a PyTorch model or Keras model, based on the arch given in the constructor.
    """

    def __init__(self, training_directory):
        self._directory = os.path.abspath(training_directory)
        self._read_model_type()

    def _fpath(self, fname):
        return os.path.join(self._directory, fname)

    def _read_model_type(self):
        """
        From the training directory, read the model type.
        If it is a dense model, this class will use PyTorch to load it,
        If it is a CNN model, it will use Keras instead.

        Throws an error if the architecture parameter is not recognized.
        """
        arch_file = self._fpath("model_identifier.txt")

        # Assert that this file exists
        assert os.path.exists(arch_file), f"File not found: {arch_file}"

        with open(arch_file, "r") as f:
            arch = f.read.strip()

        # Make sure that arch is valid, it must be "dense" or "conv"!
        assert arch in ["dense", "conv"], f"Invalid arch parameter: {arch}"
        self.arch = arch

    def get_model(self, tag: str = "latest"):
        """Loads and returns the neural network model."""
        if self.arch == "conv":
            import tensorflow as tf

            return tf.keras.models.load_model(
                os.path.join(self._directory, f"models/{tag}")
            )
        elif self.arch == "dense":
            import torch

            return torch.load(os.path.join(self._directory, "model.pt"))

        raise RuntimeError(f"Cannot load model of given architecture: {self.arch}")

    def get_sequence(self, sequence_type="training"):
        return load(self._fpath(f"{sequence_type}_sequence.pkl"))

    def get_features(self):
        return load(self._fpath("features.pkl"))

    def get_history(self):
        return load(self._fpath("history.pkl"))

    def get_feature_scaler(self):
        return load(self._fpath("feature_scaler.pkl"))
