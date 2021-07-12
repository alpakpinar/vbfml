import numpy as np
import os
from unittest import TestCase
import ROOT as r
from datagen import SingleDatasetGenerator
from array import array

def create_test_tree(filename, treename, branches, n_events, max_instances=1):
    f = r.TFile(filename, "RECREATE")
    t = r.TTree(treename, treename)
    arrays = {}

    n = array("i",[0])
    t.Branch(
            "n", 
            n,
            f'n/I'
            )
    for branch in branches:
        arr = array("d",max_instances*[0])
        t.Branch(
                branch, 
                arr,
                f'{branch}[n]/F'
                )
        arrays[branch] = arr
    for _ in range(n_events):
        if max_instances>1:
            n[0] = int(np.random.randint(low=1, high=max_instances))
        else:
            n[0] = 1
        for i in range(int(n[0])):
            for branch in branches:
                arrays[branch][i] = np.random.randn()
        t.Fill()
    f.Write()
    f.Close()

class TestSingleDatasetGen(TestCase):
    def test_single(self):
        treename = "tree"
        branches = ["a","b"]
        n_events = 10
        n_file = 2
        files = []
        total_events = n_events * n_file
        for i in range(n_file):
            fname = os.path.abspath(f"test_single_{i}.root")

            create_test_tree(
                filename=fname,
                treename=treename,
                branches=branches,
                n_events=n_events,
                max_instances=1
            )
            files.append(fname)
            # self.addCleanup(os.remove, fname)

        sdg = SingleDatasetGenerator(
            files=files,
            branches=branches,
            treename=treename,
            dataset="dataset"
        )

        # Read all events in one go
        x, y = sdg.read_events(n_events)
        assert(x.shape == (n_events, len(branches)))
        assert(y.shape == (n_events, 1))
