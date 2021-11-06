#!/usr/bin/env python

import os
import sys
import gzip
import pickle
import uproot
import numpy as np

from tqdm import tqdm
from pprint import pprint

pjoin = os.path.join


def main():
    # Path to directory containing pkl files
    inpath = sys.argv[1]
    infiles = [pjoin(inpath, f) for f in os.listdir(inpath) if f.endswith(".pkl.gz")]

    outtag = inpath.split("/")[-2]

    for infile in tqdm(infiles):
        # Decompress the file and read the pickled contents
        with gzip.open(infile, "rb") as fin:
            data = pickle.load(fin)

        # Get dataset name
        dataset_name = os.path.basename(infile).replace(".pkl.gz", "")

        if "2018" in dataset_name:
            continue

        # Scout branches
        inputs = [input for input in data.keys() if "norm" not in input]
        numevents = len(data[inputs[0]])

        # Read the normalization factor
        norm = data["normalization"]

        outdir = f"./output/{outtag}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outrootfile = pjoin(outdir, f"tree_{dataset_name}.root")

        # Save the tree
        with uproot.recreate(outrootfile) as f:
            tree_data = {}
            for inputname in inputs:
                if "nBins" in inputname:
                    tree_data[inputname] = np.stack(data[inputname]).astype(np.uint16)
                else:
                    tree_data[inputname] = np.stack(np.array(data[inputname]))

            tree_data["Normalization"] = [norm] * numevents
            f["sr_vbf"] = tree_data


if __name__ == "__main__":
    main()
