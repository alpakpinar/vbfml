import os
from unittest import TestCase

from vbfml.tests.util import create_test_tree
from vbfml.input.sequences import MultiDatasetSequence

class TestMultiDatasetSequence(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.branches = ["a", "b"]
        self.nevents_per_file = 10000
        self.n_file = 2
        self.n_features = len(self.branches)
        self.values = list(range(self.n_file))
        self.total_events = self.nevents_per_file * self.n_file
        self.dataset = "dataset"
        self.files = []
        self.batch_size = 50

        self.mds = MultiDatasetSequence(
            batch_size=self.batch_size, branches=self.branches, shuffle=False
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

    def test_n_dataset(self):
        """Test that the number of datasets is as expected"""
        self.assertEqual(len(self.mds.datasets), self.n_file)

    def test_total_events(self):
        """Test that the number of total events is as expected"""
        self.assertEqual(self.mds.total_events(), 11000)

    def test_fractions(self):
        """Test that the fraction of each dataset relative to the total events is correct."""
        self.assertAlmostEqual(self.mds.fractions["dataset_0"], 10000 / (10000 + 1000))
        self.assertAlmostEqual(self.mds.fractions["dataset_1"], 1000 / (10000 + 1000))

    def test_add_dataset_guard(self):
        """Test that add_dataset is guarded against incorrect usage"""
        # Try to add existing data set again
        with self.assertRaises(IndexError):
            self.mds.add_dataset(name="dataset_0", files=[], n_events=0)

    def test_batch_shapes(self):
        """Test that the individual batches are shaped correctly."""
        batch_indices = [0, len(self.mds) - 1]

        for idx in batch_indices:
            features, labels = self.mds[idx]
            self.assertEqual(labels.shape[1], features.shape[1])
            self.assertEqual(labels.shape[0], 1)
            self.assertEqual(features.shape[0], len(self.branches))

            self.assertTrue(features.shape[1] < self.batch_size * 1.25)
            self.assertTrue(features.shape[1] > self.batch_size / 1.25)

    def test_batch_content_noshuffle(self):
        self.mds.shuffle = False
        features, labels = self.mds[0]

        first_dataset = True
        for f1, label in zip(features[:, 0], labels[:, 0]):

            # Check that values and labels agree
            valid = False
            valid |= (f1 == 0) and (label == "dataset_0")
            valid |= (f1 == 1) and (label == "dataset_1")
            self.assertTrue(valid)

            # Since we did not shuffle, can check
            # that we never see dataset 0 again
            # after once seeing dataset 1
            if f1 == 1:
                first_dataset = False
            if not first_dataset:
                self.assertEqual(f1, 1)
