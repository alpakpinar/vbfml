import os
from copy import deepcopy
from unittest import TestCase

import numpy as np
from tensorflow.keras.utils import to_categorical
from vbfml.input.sequences import DatasetInfo, MultiDatasetSequence
from vbfml.tests.util import create_test_tree
from vbfml.models import sequential_dense_model


class TestMultiDatasetSequence(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.branches = ["a", "b"]
        self.nevents_per_file = 10000
        self.n_datasets = 2
        self.n_features = len(self.branches)
        self.values = list(range(self.n_datasets))
        self.total_events = self.nevents_per_file * self.n_datasets
        self.dataset = "dataset"
        self.files = []
        self.batch_size = 50

        self.mds = MultiDatasetSequence(
            batch_size=self.batch_size, branches=self.branches, shuffle=False
        )

        for i in range(self.n_datasets):
            name = f"dataset_{i}"
            fname = os.path.abspath(f"test_{name}.root")
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

            dataset = DatasetInfo(
                name=name,
                files=[fname],
                n_events=10000 if i == 0 else 1000,
                treename=self.treename,
            )
            self.mds.add_dataset(dataset)

    def test_n_dataset(self):
        """Test that the number of datasets is as expected"""
        self.assertEqual(len(self.mds.datasets), self.n_datasets)

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
        dataset = self.mds.get_dataset("dataset_0")
        with self.assertRaises(IndexError):
            self.mds.add_dataset(dataset)

    def test_remove_dataset(self):
        """Test removing a data set from the Sequence"""
        info = self.mds.remove_dataset("dataset_0")
        new_n_datasets = self.n_datasets - 1
        self.assertEqual(len(self.mds.datasets), new_n_datasets)
        self.assertEqual(len(self.mds.fractions), new_n_datasets)

        # The label_encoding dict stores two entries per data set
        # 1. data set name -> integer identifier
        # 2. integer identifier -> data set name
        self.assertEqual(len(self.mds.label_encoding), 2 * new_n_datasets)

        self.assertEqual(info.name, "dataset_0")

    def test_batch_shapes(self):
        """
        Test that the individual batches are shaped correctly.

        The expected shape is
        (N_batch, N_feature) for features
        (N_batch, 1) for labels
        """

        for idx in range(len(self.mds)):
            features, labels = self.mds[idx]
            # First index must agree between labels, features
            self.assertEqual(labels.shape[0], features.shape[0])

            # Batch size might vary slightly
            self.assertTrue(features.shape[0] < self.batch_size * 1.25)
            self.assertTrue(features.shape[0] > self.batch_size / 1.25)

            # Second index differs
            self.assertEqual(labels.shape[1], len(self.mds.datasets))
            self.assertEqual(features.shape[1], len(self.branches))

    def test_batch_content_noshuffle(self):
        self.mds.shuffle = False
        features, labels = self.mds[0]

        first_dataset = True

        for i_batch in range(features.shape[0]):

            x = features[i_batch, :]
            y = labels[i_batch, :]

            label_0 = to_categorical(
                self.mds.encode_label("dataset_0"), num_classes=self.n_datasets
            )
            label_1 = to_categorical(
                self.mds.encode_label("dataset_1"), num_classes=self.n_datasets
            )

            # Check that values and labels agree
            valid = False
            valid |= (x[0] == 0) and np.all(y == label_0)
            valid |= (x[0] == 1) and np.all(y == label_1)
            self.assertTrue(valid)

            # Since we did not shuffle, can check
            # that we never see dataset 0 again
            # after once seeing dataset 1
            if x[0] == 1:
                first_dataset = False
            if not first_dataset:
                self.assertEqual(x[0], 1)

    def test_batch_content_custom_dataset_label(self):
        self.mds.shuffle = False

        # Add two new datasets with different names
        # but same label
        info = deepcopy(self.mds.get_dataset("dataset_0"))
        info.name = "dataset_0_copy_1"
        info.label = "some_label"
        self.mds.add_dataset(info)

        info = deepcopy(self.mds.get_dataset("dataset_0"))
        info.name = "dataset_0_copy_2"
        info.label = "some_label"
        self.mds.add_dataset(info)

        # Remove original data sets for simplicity
        self.mds.remove_dataset("dataset_0")
        self.mds.remove_dataset("dataset_1")

        _, y = self.mds[0]
        # Check that we have exactly two data sets
        self.assertEqual(len(self.mds.datasets), 2)

        # But only one label
        # the construction with argmax is needed to undo
        # the one-hot encoding
        self.assertEqual(len(np.unique(np.argmax(y, axis=1))), 1)

    def test_keras(self):
        """
        Ensure that our output does not make keras crash. No validation of result!
        """
        model = sequential_dense_model(
            n_features=len(self.branches),
            n_layers=1,
            n_nodes=[2],
            n_classes=len(self.mds.dataset_labels()),
        )

        model.summary()
        model.fit(self.mds, epochs=1)
