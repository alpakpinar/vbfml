#!/bin/bash

# ======================================================================
#
# Script to run a set of tests as a part of the GitHub Actions workflow.
# With each pull request, this script aims to validate that the following
# functionalities are working:
#
# - Setting up a training area and running training
# - Analyzing training results
# - Pre-processing images for CNN (rotation)
#
# This script is meant to be run ONLY for GitHub Actions pipelines. 
# ======================================================================

echo "Starting test jobs: $(date)"
ls -lah .

MODEL_CONFIG_FILE=$(realpath vbfml/config/convolutional_model.yml)
ROOT_INPUT_DIR=$(realpath vbfml/.github/input_files)

# Set up and run a training area
cd vbfml/scripts

TRAINING_AREA=$(realpath output/test_job)

echo "Setting up training area from ${ROOT_INPUT_DIR}"
./train.py -d ${TRAINING_AREA} setup --input-dir ${ROOT_INPUT_DIR} --model-config ${MODEL_CONFIG_FILE}

echo "Running training for a single epoch"
./train.py -d ${TRAINING_AREA} train -n 1

# Analyze training results
echo "Analyzing training results"
./analyze_training.py analyze ${TRAINING_AREA}
./analyze_training.py plot ${TRAINING_AREA}

# Test pre-processing
echo "Testing pre-processing of images"
./preprocess_image.py rotate-all -i ${ROOT_INPUT_DIR}

# Try plotting few pre-processed images
PREPROCESSED_ROOT_DIR="${ROOT_INPUT_DIR}_preprocessed"
NUM_EVENTS_TO_PLOT=5

echo "Plotting ${NUM_EVENTS_TO_PLOT} images"
./preprocess_image.py plot-rotation-all -i ${PREPROCESSED_ROOT_DIR} -n ${NUM_EVENTS_TO_PLOT}

echo "Finished test jobs: $(date)"
