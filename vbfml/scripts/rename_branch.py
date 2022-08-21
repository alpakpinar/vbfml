#!/usr/bin/env python3
import glob
import uproot
import os
import numpy as np
import click
import pandas as pd

from math import ceil
from tqdm import tqdm

from vbfml.util import get_process_tag_from_file

pjoin = os.path.join


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-i",
    "--input-file",
    required=True,
    help="Path to the ROOT file.",
)
def rename(input_file: str):
    """
    Rename the branch of the Energy pixels (from Jetenergy_E to Jetenergy_pixels)
    """

    output_dir = f"{os.path.dirname(input_file)}_new_names/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = output_dir + os.path.basename(input_file)

    # define writable file
    file = uproot.recreate(output_file)
    print(output_file)

    # load the tree and initilize batch info
    tree = uproot.open(f"{input_file}:sr_vbf")
    batch_size = 5000
    batch_number = ceil(tree.num_entries / batch_size)
    batch_counter = 0

    # iterate by batches of 500 over the tree, aim to save memoryquit
    for sub_tree in tree.iterate(
        [],
        step_size=batch_size,
        library="np",
    ):

        batch_counter += 1
        print(f"batch {batch_counter}/{batch_number}")

        batch_index = (batch_counter - 1) * batch_size

        new_tree = {}
        for branch in tree.keys():  # copy all the other branches
            if branch == "JetImage_E" :
                branch_name = "JetImage_pixels"
            else : 
                branch_name = branch

            new_tree[branch_name] = tree[branch].arrays(
                entry_start=batch_index,
                entry_stop=batch_index + batch_size,
                library="np",
            )[branch]

        if batch_counter == 1:
            file["sr_vbf"] = new_tree
        else:
            file["sr_vbf"].extend(new_tree)

        print(file["sr_vbf"].num_entries)


@cli.command()
@click.option(
    "-i",
    "--input-dir",
    required=True,
    help="Path to the directory with the input ROOT files.",
)
def rename_all(input_dir: str):
    """
    rename all file in a directory
    """
    files = glob.glob(pjoin(input_dir, f"*root"))

    for file in tqdm(files, desc="Renaming branches"):
        print(
            f"/////////////////// renaming {os.path.basename(file)}///////////////////"
        )
        os.system("./rename_branch.py rename -i " + file)

if __name__ == "__main__":
    cli()