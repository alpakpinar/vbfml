import ROOT as r
from array import array
import numpy as np


def create_test_tree(
    filename, treename, branches, n_events, value=None, max_instances=1
):
    f = r.TFile(filename, "RECREATE")
    t = r.TTree(treename, treename)
    arrays = {}

    n_instances = array("i", [0])
    t.Branch("n", n_instances, f"n/I")
    for branch in branches:
        arrays[branch] = array("d", max_instances * [0])
        t.Branch(branch, arrays[branch], f"{branch}[n]/D")

    for _ in range(n_events):
        if max_instances > 1:
            n_instances[0] = int(np.random.randint(low=1, high=max_instances))
        else:
            n_instances[0] = 1
        for i in range(int(n_instances[0])):
            for branch in branches:
                if value is None:
                    arrays[branch][i] = np.random.randn()
                else:
                    arrays[branch][i] = value
        t.Fill()
    f.Write()
    f.Close()
