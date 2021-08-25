import os
from unittest import TestCase

import numpy as np
from vbfml.input import SingleDatasetSequence
from vbfml.tests.util import create_test_tree


class TestSingleDataseSequence(TestCase):

    def setUp(self):
        self.treename = "tree"
        self.branches = ["a","b"]
        self.nevents_per_file = 10
        self.n_file = 5
        self.n_features = len(self.branches)
        self.values = list(range(self.n_file))
        self.total_events = self.nevents_per_file * self.n_file
        self.dataset = "dataset"
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
            # self.addCleanup(os.remove, fname)

        self.sds = SingleDatasetSequence(
            files=self.files,
            branches=self.branches,
            treename=self.treename,
            dataset=self.dataset
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


    def test_read_event_features_single_file(self):
        """Read without crossing file boundaries"""
        start = 0
        stop = 5

        for file_index in range(len(self.files)):
            df = self.sds.read_event_features_single_file(file_index=file_index, local_start=start, local_stop=stop)
            
            expected_nevents = stop-start
            self.assertEqual(len(df), expected_nevents)
            self.assertEqual(len(df.columns), self.n_features)

            for column in df.columns:
                print(file_index, df[column])
                self.assertTrue(np.all(df[column]==file_index))



    def test_readevents_dimensions(self):
        """   """
        for start in range(self.total_events):
            for stop in range(start, self.total_events):
                x, y = self.sds.read_events(start=start, stop=stop)
                expected_nevents = stop-start
                self.assertEqual(x.shape[0], self.n_features)
                self.assertEqual(x.shape[1], expected_nevents)
                self.assertEqual(y.shape[0], 1)
                self.assertEqual(y.shape[1], expected_nevents)

                for i in range(x.shape[1]):
                    obs = x[0,i]
                    exp = (start + i)// self.nevents_per_file
                    print(x)
                    self.assertEqual(exp, obs, msg=f'start={start}, stop={stop}, i={i}')
                    # self.assertEqual(x[1,i], start // self.nevents_per_file)
                    # self.assertEqual(y[0,i], self.dataset)
            


    def test_index_into_file(self):
        """Translation of global event index to file index + local event index"""
        for global_event_index in range(self.total_events):
            file_index, local_event_index = self.sds._index_into_file(global_event_index)
            expected_file_index = global_event_index // self.nevents_per_file
            expected_local_event_index =  global_event_index % self.nevents_per_file
            self.assertEqual(file_index, expected_file_index)
            self.assertEqual(local_event_index,expected_local_event_index)
