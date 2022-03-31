#!/usr/bin/env python3
import glob
import uproot
import os
import numpy as np
import pandas as pd
import click
import random

from math import ceil

from vbfml.training.plot import ImagePlotter
from vbfml.util import get_process_tag_from_file

@click.group()
def cli() :
    pass

@cli.command()
@click.option(
    "-i",
    "--input-files",
    required=True,
    help="Path to the ROOT file.",
)
def rotate(input_files: str) : 
    """
    Preprocess the image (in the root file) with a phi-rotation and eta-inversion to have the leading jet in the right center
    """

    # location and name of new root file -> in a new directory "_preprocessed"
    output_dir = os.path.dirname(input_files) + "_preprocessed/"
    if not os.path.exists(output_dir) :
        os.mkdir(output_dir)
    output_file = output_dir + os.path.basename(input_files)

    # define writable file
    file = uproot.recreate(output_file)
    
    # size of the image
    n_eta_bins: int = 40
    n_phi_bins: int = 20
    im_shape = (n_eta_bins, n_phi_bins)

    # load the tree and initilize batch info
    Tree = uproot.open(input_files+":sr_vbf")
    batch_size = 5000
    batch_number = ceil(Tree.num_entries/batch_size)
    batch_counter = 0


    #iterate by batches of 500 over the tree, aim to save memoryquit
    for sub_tree in Tree.iterate(step_size = batch_size, library = "np") : 
        
        batch_counter += 1
        print(f"batch {batch_counter}/{batch_number}" )

        batch_index = (batch_counter-1)*batch_size

        # initialize the new array for the preprocess images
        New_images = np.zeros((batch_size,n_eta_bins*n_phi_bins))

        # fix size of last batch
        if (batch_counter == batch_number):
            batch_size = Tree.num_entries%batch_size
            New_images = np.zeros((batch_size,n_eta_bins*n_phi_bins))

        New_images_batch = np.ones((batch_size,n_eta_bins, n_phi_bins))
        for j in range(batch_size) :
            New_images_batch[j] = sub_tree['JetImage_pixels'][j].reshape(im_shape)

            # set phi = 0 (leading jet is at the center horizontal)
            for i in range(n_eta_bins):
                shift_phi = - round(sub_tree['leadak4_phi'][j] * n_phi_bins / (2*np.pi))
                New_images_batch[j][i] = np.roll(New_images_batch[j][i],shift_phi)
            
            # set eta > 0 (leading jet is on the right side)
            if(sub_tree['leadak4_eta'][j] < 0): 
                New_images_batch[j] = np.flip(New_images_batch[j],0)

            New_images[j] = New_images_batch[j].flatten()

        # writing in the new tree
        new_tree = {}
        new_tree["JetImage_pixels_preprocessed"] = New_images  # add the new branch
        for branch in Tree.keys() :                             # copy all the other branches
            new_tree[branch] = Tree[branch].arrays(entry_start=batch_index, entry_stop=batch_index+batch_size, library='np')[branch]

        if(batch_counter==1):
            file["sr_vbf"] = new_tree
        else :
            file["sr_vbf"].extend(new_tree)
        
        print(file["sr_vbf"].num_entries)
    

@cli.command()
@click.option(
    "-i",
    "--input-dir",
    required=True,
    help="Path to the directory with the input ROOT files.",
)
def rotate_all(input_dir: str) : 
    """
    Preprocess all root files of the directory by applying rotate function on every files of the input_dir
    """
    files = glob.glob(input_dir+"/*root")
    
    for file in files :
        print("/////////////////// \nprocessing " + os.path.basename(file) + "\n///////////////////")
        os.system('./preprocess_image.py rotate -i ' + file)


@cli.command()
@click.option(
    "-i",
    "--input-files",
    required=True,
    help="Path to the directory with the input ROOT files.",
)
@click.option(
    "-n",
    "--name-save",
    default= "plots_image_processing",
    required=False,
    help="Name of the directory where the plots are saved",
)
def plot_rotation(input_files: str, name_save: str) : 
    """
    Plot random images in the first 500 events before and after processing to check rotation
    """

    # download tree and test if it is preprocessed
    Tree = uproot.open(input_files+":sr_vbf")
    assert (
        'JetImage_pixels_preprocessed' in Tree.keys()
    ), "Root file not preprocessed ! No -JetImage_pixels_preprocessed- branch !"

    process_tag = get_process_tag_from_file(input_files)

    # location of the plots -> in a new directory "_preprocessed"
    output_dir = os.path.dirname(os.path.dirname(input_files)) + "/" + name_save
    if not os.path.exists(output_dir) :
        os.mkdir(output_dir)

    batch_size = 500

    for sub_tree in Tree.iterate(step_size = batch_size, library = "np") : 

        # load variables
        image = sub_tree['JetImage_pixels']
        image_pre = sub_tree['JetImage_pixels_preprocessed']
        eta = sub_tree['leadak4_eta']
        phi = sub_tree['leadak4_phi']
        for i in range(5):
            print(f"image{i}")

            index = random.randint(0, batch_size-1)

            # plot the image before preprocessing
            plotter = ImagePlotter()
            plotter.plot(
                image[index],
                output_dir, 
                f"image{i}", 
                vmin= 1, 
                vmax= 300, 
                left_label= f"$\eta$ = {eta[index]:.2f} // $\phi$ = {phi[index]:.2f}",
                right_label= process_tag
                )

            # plot the image after processing
            plotter = ImagePlotter()
            plotter.plot(
                image_pre[index],
                output_dir, 
                f"image{i}_preprocessed", 
                vmin= 1, 
                vmax= 300, 
                left_label= f"$\eta$ = {abs(eta[index]):.2f} // preprocessed",
                right_label= process_tag
                )

        break # reall dum but works ....

@cli.command()
@click.option(
    "-i",
    "--input-dir",
    required=True,
    help="Path to the directory with the input ROOT files.",
)
def plot_rotation_all(input_dir: str) : 
    """
    use function plot_rotation on all root files
    """
    files = glob.glob(input_dir+"/*root")
    
    for file in files :
        print("/////////////////// \nplotting " + os.path.basename(file) + "\n///////////////////")
        name_dir = os.path.basename(file)[:44]
        os.system('./preprocess_image.py plot-rotation -i ' + file + ' -n /plots_image_processing/' + name_dir)


if __name__ == "__main__":
    cli()