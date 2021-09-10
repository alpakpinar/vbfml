from vbfml.input.sequences import MultiDatasetSequence, DatasetInfo
import numpy as np
import os
from unittest import TestCase
from vbfml.training.util import load, save, get_n_events,get_weight_integral

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
        self.values = np.linspace(0, 9, self.nevents)
        self.files = []
        self.wdir = make_tmp_dir()
        self.addCleanup(os.rmdir, self.wdir)

        fname = os.path.join(self.wdir, "test.root")

        create_test_tree(
            filename=fname,
            treename=self.treename,
            branches=self.branches,
            n_events=self.nevents,
            value=self.values,
        )

        self.files.append(fname)
        self.addCleanup(os.remove, fname)

        self.mds = MultiDatasetSequence(
            batch_size=2, branches=self.branches[:-1], shuffle=False, weight_expression=self.branches[-1]
        )
        dataset = DatasetInfo(
            name='dataset',
            files=[fname],
            n_events=self.nevents,
            treename=self.treename,
        )
        self.mds.add_dataset(dataset)

    def test_get_n_events(self):
        n = get_n_events(self.files[0], self.treename)
        self.assertEqual(n, self.nevents)

    def test_get_weight_integral(self):
        weight_integrals = get_weight_integral(self.mds)
        self.assertEqual(len(weight_integrals), 1)
        self.assertTrue('dataset' in weight_integrals)
        self.assertAlmostEqual(weight_integrals['dataset'], np.sum(self.values))