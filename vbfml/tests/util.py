import os
import random
import string
import numpy as np
import uproot


def is_iterable(value):
    """Helper to determine if an object is iterable"""
    try:
        iter(value)
    except TypeError:
        return False
    return True


def create_test_tree(filename, treename, branches, n_events, value=None):
    """Streamlined dummy tree creation as input to testing suite"""
    # Determine if values are to be iterated
    iterable = is_iterable(value)
    if iterable:
        assert n_events == len(value)

    # Branch filling
    tree_data = {}
    for branch in branches:
        if value is None:
            value_out = np.random.randn((1, n_events))
        elif iterable:
            value_out = np.array(value).reshape((1, n_events))
        else:
            value_out = np.ones((1, n_events)) * value

        assert value_out.shape == (1, n_events)
        tree_data[branch] = np.array(value_out).flatten()

    with uproot.recreate(filename) as file:
        file[treename] = tree_data


def random_id(length=16):
    """Random string ID for naming temporary files etc"""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def make_tmp_dir():
    """Creation of randomly named working directories for tests"""
    wdir = "/tmp/tmp_" + random_id(32)
    if os.path.exists(wdir):
        return make_tmp_dir()
    os.makedirs(wdir)
    return wdir
