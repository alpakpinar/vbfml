#!/usr/bin/env python

import os
import sys
import re
import gzip
import pickle
import uproot
import numpy as np

from tqdm import tqdm
from pprint import pprint
from vbfml.util import vbfml_path

pjoin = os.path.join


def main():
    """
    This script transforms the input pickle files to ROOT tries via Uproot4.
    Meant to be used for transforming the event image data.

    INPUT  : .pkl files containing dictionaries which hold the image data.
    OUTPUT : .root files containing a TTree with the event data. Events will have
    the image pixels and pixel shapese saved to them as branches.
    """
    # Path to directory containing pkl files
    inpath = sys.argv[1]
    infiles = [pjoin(inpath, f) for f in os.listdir(inpath) if f.endswith(".pkl.gz")]

    outtag = inpath.split("/")[-2]

    outdir = vbfml_path(f"root/{outtag}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f'Will save ROOT files under: {outdir}')

    for infile in tqdm(infiles):
        # Decompress the file and read the pickled contents
        with gzip.open(infile, "rb") as fin:
            data = pickle.load(fin)

        # Get dataset name
        dataset_name = os.path.basename(infile).replace(".pkl.gz", "")

        if "2018" in dataset_name:
            continue

        # Scout branches
        inputs = [input for input in data.keys() if input not in ["xs", "sumw"]]
        if len(inputs) == 0:
            print(f'No input found, skipping: {dataset_name}')
            continue
        
        numevents = len(data[inputs[0]])

        # Read weight values necessary to compute normalization factor
        xs = data["xs"]
        sumw = data["sumw"]

        outrootfile = pjoin(outdir, f"tree_{dataset_name}.root")

        # Save the tree
        with uproot.recreate(outrootfile) as f:
            tree_data = {}
            for inputname in inputs:
                try:
                    if "nBins" in inputname:
                        tree_data[inputname] = np.stack(data[inputname]).astype(np.uint16)
                    else:
                        tree_data[inputname] = np.stack(np.array(data[inputname]))

                except ValueError:
                    print(f'Skipping: {inputname}')
                    continue

            tree_data["xs"] = [xs] * numevents
            tree_data["sumw"] = [sumw] * numevents

            f["sr_vbf"] = tree_data


if __name__ == "__main__":
    main()
