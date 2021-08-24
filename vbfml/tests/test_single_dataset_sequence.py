import os
from unittest import TestCase

from vbfml.input import SingleDatasetSequence
from vbfml.tests.util import create_test_tree

class TestSingleDataseSequence(TestCase):

    def setUp(self):
        self.treename = "tree"
        self.branches = ["a","b"]
        self.nevents_per_file = 10
        self.n_file = 5
        self.values = list(range(self.n_file))
        self.total_events = self.nevents_per_file * self.n_file

        self.files = []
        for i in range(self.n_file):
            fname = os.path.abspath(f"test_single_{i}.root")

            create_test_tree(
                filename=fname,
                treename=self.treename,
                branches=self.branches,
                n_events=self.nevents_per_file,
                value=self.values[i],
                max_instances=1
            )
            self.files.append(fname)
            self.addCleanup(os.remove, fname)

        self.sds = SingleDatasetSequence(
            files=self.files,
            branches=self.branches,
            treename=self.treename,
            dataset="dataset"
        )

    def test_file_list(self):
        expected_keys = list(sorted(self.files))
        observed_keys = list(sorted(self.sds.files))
        self.assertTrue(expected_keys == observed_keys)
    
    def test_nevents(self):
        self.assertTrue(len(self.sds.nevents_per_file))
        for file_index, nevents in self.sds.nevents_per_file.items():
            filepath = self.sds._file_path(file_index)
            self.assertTrue(filepath in self.files)
            self.assertTrue(nevents == self.nevents_per_file)


            