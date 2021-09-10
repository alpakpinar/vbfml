import os
from unittest import TestCase
from vbfml.training.util import load, save

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
