#!/usr/bin/env python3

import os
import sys
import uproot

from glob import glob
from pprint import pprint
from tqdm import tqdm

from vbfml.input.uproot import UprootReaderMultiFile

pjoin = os.path.join


def main():
    indir = "/eos/user/a/aakpinar/nanopost/31Mar22_jetImages/MET/MET_ver1_2017C/220401_010550/0000"

    files = glob(pjoin(indir, "nano*root"))

    branches = [
        "event",
        "Jet_pt",
        "Jet_eta",
        "Jet_hfsigmaEtaEta",
        "Jet_hfsigmaPhiPhi",
        "Jet_hfcentralEtaStripSize",
    ]

    events = [240816015, 241105675, 241068883, 240761696, 240532185]

    for file in tqdm(files, desc="Searching files"):
        t = uproot.open(file)["Events"]
        df = t.arrays(
            expressions=branches,
            library="pandas",
        )

        mask = df["event"].isin(events)
        if len(df[mask]) > 0:
            print(df[mask])


if __name__ == "__main__":
    main()
