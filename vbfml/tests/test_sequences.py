import os
from unittest import TestCase

import numpy as np
from vbfml.tests.util import create_test_tree
from vbfml.input.sequences import MultiDatasetSequence


class TestMultiDatasetSequenceNoShuffle(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.branches = ["a", "b"]
        self.nevents_per_file = 10000
        self.n_file = 3
        self.n_features = len(self.branches)
        self.values = list(range(self.n_file))
        self.total_events = self.nevents_per_file * self.n_file
        self.dataset = "dataset"
        self.files = []

        self.mds = MultiDatasetSequence(
            batch_size=50, branches=self.branches, shuffle=False
        )

        for i in range(self.n_file):
            label = f"dataset_{i}"
            fname = os.path.abspath(f"test_{label}.root")
            create_test_tree(
                filename=fname,
                treename=self.treename,
                branches=self.branches,
                n_events=self.nevents_per_file,
                value=self.values[i],
                max_instances=1,
            )
            self.files.append(fname)
            self.addCleanup(os.remove, fname)

            self.mds.add_dataset(
                name=label, files=[fname], n_events=10000 if i == 0 else 1000
            )

    def test_empty(self):
        pass
