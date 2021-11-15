from dataclasses import dataclass

import os
import yaml
import vbfml
import pandas as pd

pjoin = os.path.join


def vbfml_path(path):
    """Returns the absolute path for the given path."""
    return pjoin(vbfml.__path__[0], path)


@dataclass
class YamlLoader:
    infile: str

    def load(self) -> dict:
        """Loads and returns data from a YAML file."""
        with open(self.infile, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data


@dataclass
class ModelConfiguration:
    """
    Object used to read model configuration data, given a .yml configuration file as an input.

    After the object is initiated, one can get any feature by calling the get() method:
    >>> mconfig = ModelConfiguration("model.yml")
    >>> my_feature = mconfig.get("my_feature")

    If the feature is not recognized, get() will raise an AssertionError.
    """

    infile: str
    data: dict = None
    model: str = None

    def __post_init__(self):
        with open(self.infile, "r") as f:
            self.data = yaml.load(f, Loader=yaml.FullLoader)
            self._set_model_arch()

    def _set_model_arch(self):
        """Set model architecture based on which config file has been passed."""
        self.model = self.data["architecture"]

    def _check_feature_is_valid(self, feature):
        """Internal function to check if the feature name for a given model is valid"""
        assert (
            feature in self.data.keys()
        ), f"Feature: {feature} is not recognized for {self.model}"

    def get(self, feature):
        """Getter for a specific feature of a specific model."""
        self._check_feature_is_valid(feature)
        return self.data[feature]


@dataclass
class MultiBatchBuffer:
    df: pd.DataFrame = None
    batch_size: int = 1
    min_batch: int = -1
    max_batch: int = -1

    def set_multibatch(self, df: pd.DataFrame, min_batch: int):
        self.df = df
        self.min_batch = min_batch
        self.max_batch = min_batch + len(df) // self.batch_size

    def __contains__(self, batch_index):
        if self.df is None:
            return False
        if not len(self.df):
            return False
        if batch_index < 0:
            return False
        return self.min_batch <= batch_index <= self.max_batch

    def clear(self):
        self.df = None
        self.min_batch = -1
        self.max_batch = -1

    def get_batch_df(self, batch_index):
        if not batch_index in self:
            raise IndexError(f"Batch index '{batch_index}' not in current buffer.")

        row_start = batch_index - self.min_batch
        row_stop = min(row_start + self.batch_size, len(self.df))
        return self.df.loc[row_start:row_stop]
