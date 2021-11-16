import os
from unittest import TestCase
from vbfml.util import vbfml_path, ModelConfiguration

pjoin = os.path.join


class TestConfigParser(TestCase):
    def setUp(self) -> None:
        self.mconfigs = []
        config_dir = vbfml_path("config")

        # We'll test each configuration file
        config_files = [pjoin(config_dir, f) for f in os.listdir(config_dir)]
        for cf in config_files:
            self.mconfigs.append(ModelConfiguration(cf))

        # Minimally, we'd like to have these features in the config file
        # (with the given data types!)
        self.features_to_check = {
            "features": list,
            "train_size": float,
            "batch_size_train": int,
            "batch_size_val": int,
            "batch_buffer_size_train": int,
            "batch_buffer_size_val": int,
        }

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
