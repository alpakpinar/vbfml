#!/usr/bin/env python3
import glob
import pickle
import uproot
import os
import re
import numpy as np
import click
import random
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak

from math import ceil
from tqdm import tqdm

from vbfml.training.plot import ImagePlotter
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
@click.option(
    "-b",
    "--image-branch",
    required=False,
    default="JetImageFine_E",
    help="Name of the branch containing image data.",
)
def rotate(input_file: str, image_branch: str):
    """
    Preprocess the image (in the root file) with a phi-rotation and eta-inversion to have the leading jet in the right center
    """

    # location and name of new root file -> in a new directory "_preprocessed"
    output_dir = f"{os.path.dirname(input_file)}_preprocessed/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = output_dir + os.path.basename(input_file)

    # Output file
    file = uproot.recreate(output_file)

    # Load the tree 
    tree = uproot.open(f"{input_file}:sr_vbf")
    
    # Determine size of the image (for reshaping 1D array into 2D)
    table_name = image_branch.split('_')[0]
    n_eta_bins = int( tree[f'{table_name}_nEtaBins'].array()[0] )
    n_phi_bins = int( tree[f'{table_name}_nPhiBins'].array()[0] )
    im_shape = (n_eta_bins, n_phi_bins)

    # Initilize batch info
    batch_size = 5000
    batch_number = ceil(tree.num_entries / batch_size)
    batch_counter = 0

    # iterate by batches of 500 over the tree, aim to save memoryquit
    for sub_tree in tree.iterate(
        [image_branch, "leadak4_phi", "leadak4_eta"],
        step_size=batch_size,
        library="np",
    ):

        batch_counter += 1
        print(f">> Processing batch {batch_counter}/{batch_number}", end="\r")

        batch_index = (batch_counter - 1) * batch_size

        # initialize the new array for the preprocess images
        new_images = np.zeros((batch_size, n_eta_bins * n_phi_bins), dtype="uint8")

        # fix size of last batch
        if batch_counter == batch_number:
            batch_size = tree.num_entries % batch_size
            new_images = np.zeros((batch_size, n_eta_bins * n_phi_bins), dtype="uint8")

        new_images_batch = np.ones((batch_size, n_eta_bins, n_phi_bins))
        for j in range(batch_size):
            new_images_batch[j] = sub_tree[image_branch][j].reshape(im_shape)

            # set phi = 0 (leading jet is at the center horizontal)
            for i in range(n_eta_bins):
                shift_phi = -round(
                    sub_tree["leadak4_phi"][j] * n_phi_bins / (2 * np.pi)
                )
                new_images_batch[j][i] = np.roll(new_images_batch[j][i], shift_phi)

            # Set leading jet eta > 0 (leading jet is on the right side)
            if sub_tree["leadak4_eta"][j] < 0:
                new_images_batch[j] = np.flip(new_images_batch[j], 0)

            # Flatten to 1D array, so that we can write it to output ROOT file
            new_images[j] = new_images_batch[j].flatten()

        # Writing in the new tree
        new_tree = {}
        new_tree[f"{image_branch}_preprocessed"] = new_images  # Add the new branch
        for branch in tree.keys():  # Copy all the other branches
            new_tree[branch] = tree[branch].arrays(
                entry_start=batch_index,
                entry_stop=batch_index + batch_size,
                library="np",
            )[branch]

        if batch_counter == 1:
            file["sr_vbf"] = new_tree
        else:
            file["sr_vbf"].extend(new_tree)

    print(f">> Done, processed {file['sr_vbf'].num_entries} entries.")


@cli.command()
@click.option(
    "-i",
    "--input-dir",
    required=True,
    help="Path to the directory with the input ROOT files.",
)
@click.option(
    "-b",
    "--image-branch",
    required=False,
    default="JetImageFine_E",
    help="Name of the branch containing image data.",
)
def rotate_all(input_dir: str, image_branch: str):
    """
    Preprocess all root files of the directory by applying rotate function on every files of the input_dir
    Warning -> have weird behavior over a lot of files. miss some events
    """
    files = glob.glob(pjoin(input_dir, f"*root"))

    print("Rotating images...")
    for file in files:
        cmd = f"./preprocess_image.py rotate -i {file} -b {image_branch}"
        print(f"\n{cmd}\n")
        os.system(cmd)


@cli.command()
@click.option(
    "-i",
    "--input-file",
    required=True,
    help="Path to the ROOT file.",
)
@click.option(
    "-n",
    "--name-save",
    default="plots_image_processing_test",
    required=False,
    help="Name of the directory where the plots are saved",
)
@click.option(
    "--ievent",
    default=None,
    type=int,
    required=False,
    help="number of the event to plot",
)
@click.option(
    "--num-events",
    default=5,
    type=int,
    required=False,
    help="number of event to plot",
)
@click.option(
    "-b",
    "--image-branch",
    required=False,
    default="JetImageFine_E",
    help="Name of the branch containing image data.",
)
def plot_rotation(input_file: str, name_save: str, ievent: int, num_events: int, image_branch: str):
    """
    Plot random images in the first 500 events before and after processing to check rotation
    """
    # download tree and test if it is preprocessed and get the channel of the process
    tree = uproot.lazy(f"{input_file}:sr_vbf")
    process_tag = get_process_tag_from_file(input_file)

    # location of the plots -> in a new directory
    outtag = os.path.basename(os.path.dirname(input_file))
    output_dir = pjoin("output", outtag, name_save)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Number of images to plot
    num_images = 1 if ievent else num_events

    # Determine size of the image (for reshaping 1D array into 2D)
    table_name = image_branch.split('_')[0]
    n_eta_bins = int( tree[f'{table_name}_nEtaBins'][0] )
    n_phi_bins = int( tree[f'{table_name}_nPhiBins'][0] )

    for i in range(num_images):

        index = ievent - 1 if ievent else random.randint(0, len(tree["mjj"]) - 1)
        print(f">> Plotting image_{index+1}")

        eta = tree["leadak4_eta", index]
        phi = tree["leadak4_phi", index]
        # plot the image before preprocessing
        plotter = ImagePlotter(n_eta_bins=n_eta_bins, n_phi_bins=n_phi_bins)
        plotter.plot(
            ak.to_numpy(tree[image_branch, index]),
            output_dir,
            f"image_{os.path.basename(input_file)[5:-5]}_{index+1}",
            vmin=1,
            vmax=255,
            left_label=f"$\eta$ = {eta:.2f} // $\phi$ = {phi:.2f}",
            right_label=process_tag,
        )

        # plot the image after processing
        plotter = ImagePlotter(n_eta_bins=n_eta_bins, n_phi_bins=n_phi_bins)
        plotter.plot(
            ak.to_numpy(tree[f"{image_branch}_preprocessed", index]),
            output_dir,
            f"image_{os.path.basename(input_file)[5:-5]}_{index+1}_preprocessed",
            vmin=1,
            vmax=255,
            left_label=f"$\eta$ = {abs(eta):.2f} (Pre-processed)",
            right_label=process_tag,
        )


@cli.command()
@click.option(
    "-i",
    "--input-dir",
    required=True,
    help="Path to the directory with the input ROOT files.",
)
@click.option(
    "-b",
    "--image-branch",
    required=False,
    default="JetImageFine_E",
    help="Name of the branch containing image data.",
)
def plot_rotation_all(input_dir: str, image_branch: str):
    """
    Use plot_rotation function on all root files within the input directory.
    """
    files = glob.glob(pjoin(input_dir, f"*root"))

    output_dir = pjoin(os.path.dirname(input_dir), "image_processing_plots")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file in files:
        subdir = re.findall('tree_(.*)\.root', os.path.basename(file))[0]
        outdir = pjoin(os.path.basename(output_dir), subdir)
        
        cmd = f"./preprocess_image.py plot-rotation -i {file} -n {outdir} -b {image_branch}"
        print(f"\n{cmd}\n")
        os.system(cmd)


@cli.command()
@click.option(
    "-i",
    "--input-file",
    required=True,
    help="Path to the ROOT file.",
)
@click.option(
    "--start",
    default=0,
    required=False,
    type=int,
    help="begin the check at event # start",
)
@click.option(
    "--stop",
    default=None,
    required=False,
    type=int,
    help="stop the check at event # stop",
)
@click.option(
    "-n",
    "--name-save",
    default="MET_distribution",
    required=False,
    help="Name of the directory where the MET are saved",
)
def check_met(input_file: str, name_save: str, start: int, stop: int):
    """
    compute the MET distribution with old image and the preprocessed image to check everything is still correct
    check if there is a 0 MET and if the images are in good format 'uint8'
    """
    # Configure output directory to save the cache with MET histograms
    outtag = os.path.basename(os.path.dirname(input_file))
    output_dir = pjoin("output", outtag, name_save)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # size of the image
    n_eta_bins, n_phi_bins = 40, 20
    im_shape = (n_eta_bins, n_phi_bins)
    eta_range = (-5, 5)
    phi_range = (-np.pi, np.pi)

    eta_centers = np.linspace(
        eta_range[0], eta_range[1], n_eta_bins, endpoint=False
    ) + (eta_range[1] - eta_range[0]) / (2 * n_eta_bins)
    phi_centers = np.linspace(
        phi_range[0], phi_range[1], n_phi_bins, endpoint=False
    ) + (phi_range[1] - phi_range[0]) / (2 * n_phi_bins)

    # load the tree and initilize batch info
    tree = uproot.open(f"{input_file}:sr_vbf")
    batch_size = 5000
    batch_counter = 0

    # write MET distribution in .pkl file
    dataset_name = (
        os.path.basename(input_file).replace("tree_", "").replace(".root", "")
    )
    cache = pjoin(output_dir, f"{dataset_name}_met_distribution.pkl")

    # Configure start and stop indices, if specific indices are given
    if start or stop:
        if not stop:
            stop = tree.num_entries
        cache = f"{cache.replace('.pkl','')}_{start}to{stop}.pkl"

    if not stop:
        stop = tree.num_entries

    number_event = stop - start
    batch_number = ceil(number_event / batch_size)

    # initialize the new array for the MET distribution
    met = {
        "JetImage_pixels": np.zeros(number_event),
        "JetImage_pixels_preprocessed": np.zeros(number_event),
    }

    # iterate by batches of 500 over the tree, aim to save memoryquit
    for sub_tree in tree.iterate(
        [
            "JetImage_pixels_preprocessed",
            "JetImage_pixels",
        ],
        entry_start=start,
        entry_stop=stop,
        step_size=batch_size,
        library="np",
    ):

        batch_counter += 1
        print(f"batch {batch_counter}/{batch_number}")

        batch_index = (batch_counter - 1) * batch_size

        # fix size of last batch
        if (batch_counter == batch_number) and (number_event % batch_size != 0):
            batch_size_iterate = number_event % batch_size
        else:
            batch_size_iterate = batch_size

        # check the format of the array to be 'uint8'
        if sub_tree["JetImage_pixels_preprocessed"].dtype != "uint8":
            print(
                f"Warning ! event {(batch_counter-1)*batch_size + j+1} doesn't have format uint8"
            )

        for j in range(batch_size_iterate):

            a, b = 0, 0

            for label in met:
                image = sub_tree[label][j].reshape(im_shape)
                energy, energy_x, energy_y = 0, 0, 0

                # compute the MET from the image
                for i in range(n_eta_bins):
                    for k in range(n_phi_bins):
                        energy_x += (
                            image[i][k]
                            * np.cos(phi_centers[k])
                            / np.cosh(eta_centers[i])
                        )
                        energy_y += (
                            image[i][k]
                            * np.sin(phi_centers[k])
                            / np.cosh(eta_centers[i])
                        )

                energy = np.sqrt(np.square(energy_x) + np.square(energy_y))

                if label == "JetImage_pixels":
                    a = energy
                else:
                    b = energy

                met[label][(batch_counter - 1) * batch_size + j] = energy

            if (a == 0) or (b == 0):
                print(
                    f"Event {(batch_counter-1)*batch_size + j+1} of file has 0 MET (empty picture maybe)"
                )

            if round(a, 6) != round(b, 6):
                print(
                    f"event {(batch_counter-1)*batch_size + j+1} does not match ({a},{b} )"
                )

    with open(cache, "wb") as f:
        pickle.dump(met, f)


@cli.command()
@click.option(
    "-i",
    "--input-dir",
    required=True,
    help="Path to the directory with the pkl files",
)
def plot_met(input_dir: str):
    """
    plot the distribution of the MET for unprepro and prepro images. plot as well there differences to see if they match
    """

    files = glob.glob(pjoin(input_dir, f"*pkl"))

    met_df = pd.DataFrame()

    for file in files:
        with open(file, "rb") as f:
            met_new = pickle.load(f)
            if not (met_new["JetImage_pixels"].all()):
                print(f"look at file {file}")
        met_df = pd.concat([met_df, pd.DataFrame(met_new)])

    n_bins = 60
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].set_xlabel("MET_normal")
    plt.suptitle("MET distribution for normal vs processed images")
    axs[0].hist(met_df["JetImage_pixels"], bins=n_bins)
    axs[1].set_xlabel("MET_processed")
    axs[0].set_ylabel("# events")
    axs[1].hist(met_df["JetImage_pixels_preprocessed"], bins=n_bins)
    plt.savefig(pjoin(input_dir, f"met_distribution.pdf"))

    diff = met_df["JetImage_pixels_preprocessed"] - met_df["JetImage_pixels"]
    fig, ax = plt.subplots(figsize=(3, 4))
    plt.suptitle("MET difference (processed-normal)")
    plt.hist(diff, bins=30)
    plt.xlabel("$\Delta$ MET")
    plt.ylabel("# events")
    plt.savefig(pjoin(input_dir, f"met_distribution_diff.pdf"))


if __name__ == "__main__":
    cli()
