import ROOT as r
from array import array
import numpy as np


def create_test_tree(filename, treename, branches, n_events, value=None):
    f = r.TFile(filename, "RECREATE")
    t = r.TTree(treename, treename)
    arrays = {}

    # Branch creation
    for branch in branches:
        arrays[branch] = array("d", [0])
        t.Branch(branch, arrays[branch], f"{branch}/D")

    # Determine if values are to be iterated
    iterable = True
    try:
        iter(value)
    except TypeError:
        iterable = False
    if iterable:
        assert n_events == len(value)

    # Branch filling
    for i_event in range(n_events):
        for branch in branches:
            if value is None:
                value_out = np.random.randn()
            elif iterable:
                value_out = value[i_event]
            else:
                value_out = value

            arrays[branch][0] = value_out
        t.Fill()
    f.Write()
    f.Close()
