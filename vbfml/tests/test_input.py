import os
from unittest import TestCase

import numpy as np

from vbfml.tests.util import create_test_tree, make_tmp_dir
from vbfml.training.input import (
    build_sequence,
    dataset_from_file_name_bucoffea,
    load_datasets_bucoffea,
)


class TestInput(TestCase):
    def test_dataset_from_file_name_bucoffea(self):
        validation_pairs = []
        for process in ["zjet", "wjet", "akajsdbk"]:
            for year in [2016, 2017, 2018, 9999]:
                dataset = f"{process}_{year}"
                file_name = f"tree_{dataset}.root"
                validation_pairs.append((file_name, dataset))

        for input_value, expected_output_value in validation_pairs:
            self.assertEqual(
                dataset_from_file_name_bucoffea(input_value), expected_output_value
            )
            self.assertEqual(
                dataset_from_file_name_bucoffea("/a/b/c/" + input_value),
                expected_output_value,
            )
            self.assertEqual(
                dataset_from_file_name_bucoffea("/" + input_value),
                expected_output_value,
            )


class TestLoadDatasetsBucoffea(TestCase):
    def setUp(self):
        self.wdir = make_tmp_dir()
        self.addCleanup(os.rmdir, self.wdir)

        files = [
            os.path.join(self.wdir, "tree_firstdataset_2017.root"),
            os.path.join(self.wdir, "tree_seconddataset_2017.root"),
        ]

        for file in files:
            create_test_tree(
                filename=file,
                treename="sr_vbf",
                branches=["a", "b", "weight_total", "xs", "sumw"],
                n_events=10,
                value=2,
            )
            self.addCleanup(os.remove, file)

    def test_load_datasets_bucoffea(self):
        """Test creation of DatasetInfo objects from a directory of ROOT files"""
        datasets = load_datasets_bucoffea(self.wdir)
        self.assertEqual(len(datasets), 2)
        dataset_names = sorted([x.name for x in datasets])
        self.assertEqual(dataset_names[0], "firstdataset_2017")
        self.assertEqual(dataset_names[1], "seconddataset_2017")
        self.assertEqual(datasets[0].n_events, 10)
        self.assertEqual(datasets[1].n_events, 10)

    def test_build_sequence(self):
        """Test automated sequence generation from data sets"""
        datasets = load_datasets_bucoffea(self.wdir)
        sequence = build_sequence(datasets, features=["a", "b"])
        features, labels, weights = sequence[0]
        self.assertEqual(features.shape[1], 2)
        self.assertTrue(np.all(weights == 2))
