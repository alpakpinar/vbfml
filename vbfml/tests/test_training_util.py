import os
from unittest import TestCase

import numpy as np

from vbfml.input.sequences import DatasetInfo, MultiDatasetSequence
from vbfml.training.util import (
    get_n_events,
    get_weight_integral_by_label,
    load,
    normalize_classes,
    save,
)

from .util import create_test_tree, make_tmp_dir, random_id


class TestSimpleUtils(TestCase):
    def setUp(self):
        self.wdir = make_tmp_dir()
        self.addCleanup(os.rmdir, self.wdir)

    def _test_load_save(self, object):
        fpath = os.path.join(self.wdir, f"{random_id()}.pkl")
        self.addCleanup(os.remove, fpath)
        save(object, fpath)
        readback = load(fpath)
        self.assertEqual(
            object, readback, msg=f"Readback failes for object: '{object}'"
        )

    def test_load_save(self):
        objects = [{}, {"a": 1}, [1, 2, 5], "hello"]
        for object in objects:
            self._test_load_save(object)


class TestClassNormalization(TestCase):
    def setUp(self):
        self.treename = "tree"
        self.branches = ["a", "weight"]
        self.nevents = 10
        self.n_features = len(self.branches)
        self.files = []
        self.wdir = make_tmp_dir()
        self.addCleanup(os.rmdir, self.wdir)

        self.mds = MultiDatasetSequence(
            batch_size=4,
            branches=self.branches[:-1],
            shuffle=False,
            weight_expression=self.branches[-1],
        )

        # Generate three data sets
        self.values = [
            np.linspace(0, 9, 10),
            np.linspace(10, 14, 5),
            np.linspace(30, 45, 15),
        ]
        self.files = [os.path.join(self.wdir, f"test_{i}.root") for i in range(3)]
        for i in range(3):
            fname = self.files[i]
            values = self.values[i]
            nevents = len(values)

            create_test_tree(
                filename=fname,
                treename=self.treename,
                branches=self.branches,
                n_events=nevents,
                value=values,
            )

            self.addCleanup(os.remove, fname)

            # The first data set is a class by itself
            # all other data sets are merged into one class
            name = f"dataset_{i}"
            if i == 0:
                label = name
            else:
                label = "merged_label"

            dataset = DatasetInfo(
                name=name,
                label=label,
                files=[fname],
                n_events=nevents,
                treename=self.treename,
            )
            self.mds.add_dataset(dataset)

    def test_get_n_events(self):
        """Test that the get_n_events correctly returns the number of events in a TTree on disk"""
        for fname, values in zip(self.files, self.values):
            readback = get_n_events(fname, self.treename)
            self.assertEqual(readback, len(values))

    def test_get_weight_integral_by_label(self):
        """Test that the get_weight_integral_by_label function correctly calculates weight sums per data set label"""
        weight_integrals = get_weight_integral_by_label(self.mds)

        self.assertEqual(len(weight_integrals), 2)
        self.assertTrue("dataset_0" in weight_integrals)
        self.assertTrue("merged_label" in weight_integrals)

        # The first class is just the first data set
        self.assertAlmostEqual(weight_integrals["dataset_0"], np.sum(self.values[0]))

        # The second class is the merged version of the second and third data sets
        self.assertAlmostEqual(
            weight_integrals["merged_label"],
            np.sum(self.values[1]) + np.sum(self.values[2]),
            places=1,
        )

    def test_normalize_classes(self):
        """Test that the normalize_classes function produces correct weight integrals"""
        # Before normalizing, the class integrals disagree
        label_to_weight = get_weight_integral_by_label(self.mds)
        self.assertNotAlmostEqual(
            label_to_weight["merged_label"], label_to_weight["dataset_0"], places=1
        )

        # Normalize them
        normalize_classes(self.mds)
        label_to_weight = get_weight_integral_by_label(self.mds)

        # Now they should agree to unity
        self.assertAlmostEqual(label_to_weight["dataset_0"], 1, places=1)
        self.assertAlmostEqual(label_to_weight["merged_label"], 1, places=1)
