import os
from unittest import TestCase

from vbfml.tests.util import make_tmp_dir, create_test_tree
from vbfml.training.input import dataset_from_file_name_bucoffea, load_datasets_bucoffea


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
                branches=["a", "b"],
                n_events=10,
                value=1,
            )
            self.addCleanup(os.remove, file)

    def test_load_datasets_bucoffea(self):
        datasets = load_datasets_bucoffea(self.wdir)
        self.assertEqual(len(datasets), 2)
