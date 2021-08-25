import os
from unittest import TestCase

import numpy as np
import uproot

from .util import create_test_tree


class TestCreateTestTree(TestCase):

    def setUp(self):
        self.treename = "tree"
        self.branches = ["a","b"]
        self.nevents_per_file = 10
        self.n_file = 1
        self.n_features = len(self.branches)
        self.values =[1337]
        self.total_events = self.nevents_per_file * self.n_file
        self.dataset = "dataset"
        self.files = []
        for i in range(self.n_file):
            fname = os.path.abspath(f"test_util_{i}.root")

            create_test_tree(
                filename=fname,
                treename=self.treename,
                branches=self.branches,
                n_events=self.nevents_per_file,
                value=self.values[i],
                max_instances=1
            )
            self.files.append(fname)
            # self.addCleanup(os.remove, fname)

    def test_file_existence(self):
        for file in self.files:
            self.assertTrue(os.path.exists(file))

    def test_tree_existence(self):
        for file in self.files:
            f = uproot.open(file)
            self.assertTrue(self.treename in f)

    def test_branch_existence(self):
        for file in self.files:
            f = uproot.open(file)
            tree = f[self.treename]
            for branch in self.branches:
                self.assertTrue(branch in tree)
        
    def test_branch_content(self):
        for i, file in enumerate(self.files):
            f = uproot.open(file)
            tree =  f[self.treename]
            df = tree.arrays(
                    expressions = self.branches,
                    library='pandas'
                )
            for branch in self.branches:
                observed_values = list(df[branch])
                expected_values = [self.values[i]] * self.nevents_per_file
                self.assertListEqual(expected_values, observed_values)
