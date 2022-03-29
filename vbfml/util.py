from dataclasses import dataclass
from typing import Dict
from collections import OrderedDict

import os
import re
import yaml
import vbfml
import subprocess
import pandas as pd


from vbfml.models import sequential_convolutional_model, sequential_dense_model

pjoin = os.path.join


def vbfml_path(path):
    """Returns the absolute path for the given path."""
    return pjoin(vbfml.__path__[0], path)


def git_rev_parse():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")


def git_diff():
    return subprocess.check_output(["git", "diff"]).decode("utf-8")


def git_diff_staged():
    return subprocess.check_output(["git", "diff", "--staged"]).decode("utf-8")


def get_process_tag_from_file(filename: str) -> str:
    """
    Given a ROOT filename, return the process tag showing
    the ground truth for this process (e.g. EWK Z(vv)).
    """
    basename = os.path.basename(filename)
    mapping = {
        ".*VBF_HToInv.*M125.*": "VBF Hinv",
        ".*EWKZ2Jets.*ZToNuNu.*": "EWK Zvv",
        ".*EWKW(Minus|Plus)2Jets.*WToLNu.*": "EWK Wlv",
        ".*Z\dJetsToNuNu.*PtZ.*": "QCD Zvv",
        ".*WJetsToLNu_Pt.*": "QCD Wlv",
    }

    for regex, label in mapping.items():
        if re.match(regex, basename):
            return label

    raise RuntimeError(f"Could not find a process tag for file: {basename}")


@dataclass
class YamlLoader:
    infile: str

    def load(self) -> dict:
        """Loads and returns data from a YAML file."""
        with open(self.infile, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data


@dataclass
class DatasetAndLabelConfiguration:
    """
    Object used to read dataset and label configuration from a given configuration file.

    After the object is initiated, dataset regex and corresponding labels can be
    retrieved by calling get_datasets():
    >>> mdataset = DatasetAndLabelConfiguration("datasets.yml")
    >>> dataset_labels = mdataset.get_datasets()
    """

    infile: str
    data: dict = None

    def __post_init__(self) -> None:
        with open(self.infile, "r") as f:
            self.data = yaml.load(f, Loader=yaml.FullLoader)

        for key in ["datasets", "scales"]:
            assert (
                key in self.data.keys()
            ), f"Missing key from in dataset configuration {self.infile}: '{key}'"

    def get_dataset_labels(self) -> Dict[str, str]:
        mapping = {}
        datasets = self.data["datasets"]
        labels = datasets.keys()
        for label in labels:
            mapping[label] = datasets[label]["regex"]
        return mapping

    def get_dataset_scales(self) -> Dict[str, float]:
        """
        Get the dataset scales corresponding to each process.
        Returns a dictionary that maps:

        { Dataset regex : Scale factor }
        """
        mapping = {}
        scales = self.data["scales"]
        labels = scales.keys()
        for label in labels:
            regex, scale = scales[label]["regex"], scales[label]["scale"]
            mapping[regex] = scale
        return mapping


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

    def __post_init__(self) -> None:
        with open(self.infile, "r") as f:
            self.data = yaml.load(f, Loader=yaml.FullLoader)

        self._set_model_arch()
        self._set_n_classes()

    def _set_n_classes(self) -> None:
        """
        Based on the dataset configuration we have, determine n_classes
        parameter at runtime.
        """
        d_config_path = vbfml_path("config/datasets/datasets.yml")
        assert os.path.exists(
            d_config_path
        ), f"Cannot look up the dataset configuration from file: {d_config_path}"
        dataset_labels = DatasetAndLabelConfiguration(
            d_config_path
        ).get_dataset_labels()
        self.data["arch_parameters"]["n_classes"] = len(dataset_labels)

    def _set_model_arch(self) -> None:
        """Set model architecture based on which config file has been passed."""
        self.model = self.data["architecture"]

    def _check_feature_is_valid(self, feature: str) -> None:
        """Internal function to check if the feature name for a given model is valid"""
        assert (
            feature in self.data.keys()
        ), f"Feature: {feature} is not recognized for {self.model}"

    def get(self, feature: str):
        """Getter for a specific feature of a specific model."""
        self._check_feature_is_valid(feature)
        return self.data[feature]


@dataclass
class ModelFactory:
    """
    Factory object used to build neural network models with different architectures.

    To build a model, one can use the build() method of this class, providing the
    ModelConfiguration object as an input:
    >>> mconfig = ModelConfiguration("config.yml")
    >>> ModelFactory.build(mconfig)
    """

    @classmethod
    def build(cls, model_config: ModelConfiguration):
        """Build the model given the ModelConfiguration object.

        Args:
            model_config (ModelConfiguration): ModelConfiguration object for the model being built.
        """
        # The type of model we want to build (e.g. Convolutional, dense etc.)
        model = model_config.get("architecture")

        # The set of parameters specifying the model architecture
        # as specified in the .yml config files
        arch_parameters = model_config.get("arch_parameters")

        builder_function = {
            "dense": sequential_dense_model,
            "conv": sequential_convolutional_model,
        }

        assert model in builder_function.keys(), f"Model {model} not recognized"

        if model == "dense":
            arch_parameters["n_features"] = len(model_config.get("features"))

        return builder_function[model](**arch_parameters)


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

        if len(df) % self.batch_size > 0:
            self.max_batch += 1

    def __contains__(self, batch_index):
        if self.df is None:
            return False
        if not len(self.df):
            return False
        if batch_index < 0:
            return False
        return self.min_batch <= batch_index < self.max_batch

    def clear(self):
        self.df = None
        self.min_batch = -1
        self.max_batch = -1

    def get_batch_df(self, batch_index):
        if not batch_index in self:
            raise IndexError(f"Batch index '{batch_index}' not in current buffer.")

        row_start = (batch_index - self.min_batch) * self.batch_size
        row_stop = min(row_start + self.batch_size, len(self.df))
        return self.df.iloc[row_start:row_stop]
