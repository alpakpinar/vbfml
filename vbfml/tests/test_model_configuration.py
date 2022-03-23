import os
from unittest import TestCase
from vbfml.util import (
    vbfml_path,
    ModelConfiguration,
    YamlLoader,
    DatasetAndLabelConfiguration,
)

pjoin = os.path.join


class TestConfigParser(TestCase):
    def setUp(self) -> None:
        self.mconfigs = []
        config_dir = vbfml_path("config")

        # We'll test each configuration file
        config_files = [
            pjoin(config_dir, f) for f in os.listdir(config_dir) if f.endswith(".yml")
        ]
        for cf in config_files:
            self.mconfigs.append(ModelConfiguration(cf))

        # Minimally, we'd like to have these features in the config file
        # (with the given data types!)
        self.features_to_check = {
            "features": list,
            "training_parameters": dict,
            "validation_parameters": dict,
            "architecture": str,
            "arch_parameters": dict,
        }

        d_config_path = vbfml_path("config/datasets/datasets.yml")
        dataset_config = DatasetAndLabelConfiguration(d_config_path)
        self.dataset_labels = dataset_config.get_datasets()

    def test_exceptions(self):
        with self.assertRaises(AssertionError):
            for mconfig in self.mconfigs:
                mconfig.get("NonExistentFeature")

    def test_feature_names(self):
        for mconfig in self.mconfigs:
            for key in self.features_to_check.keys():
                self.assertIn(key, mconfig.data)

    def test_return_types(self):
        for mconfig in self.mconfigs:
            for key, type in self.features_to_check.items():
                self.assertIsInstance(mconfig.get(key), type)

    def test_n_classes(self):
        """
        Check if n_classes is set properly according to the dataset configuration.
        """
        feature_name = "n_classes"
        for mconfig in self.mconfigs:
            self.assertIn(feature_name, mconfig.data["arch_parameters"])

            n_classes = mconfig.data["arch_parameters"]["n_classes"]
            self.assertEqual(len(self.dataset_labels), n_classes)


class ParamGridParser(TestCase):
    def setUp(self) -> None:
        grid_search_dir = vbfml_path("config/gridsearch")
        grid_file = pjoin(grid_search_dir, os.listdir(grid_search_dir)[0])

        loader = YamlLoader(grid_file)
        self.grid = loader.load()

    def test_grid_branch(self):
        keys = list(self.grid.keys())
        # Has to be one branch in this file
        self.assertEqual(len(keys), 1)
        # And it has to be named as "param_grid"
        self.assertEqual(keys[0], "param_grid")

    def test_branch_types(self):
        keys = list(self.grid.keys())
        param_grid = self.grid[keys[0]]
        # Each branch must be a list
        for k, v in param_grid.items():
            self.assertIsInstance(v, list)
