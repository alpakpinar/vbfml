#!/bin/bash

# Wrapper script for job execution on HTCondor clusters

echo "Starting: $(date)"
echo "Running on: $(hostname)"
echo "uname -a: $(uname -a)"

ARGS=("$@")

# Source Python 3.9 environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_101swan/x86_64-centos7-gcc8-opt/setup.sh
echo "Using python at: $(which python)"
python --version

JOB_DIR=${PWD}
echo "Job directory: ${JOB_DIR}"

# Extract the repo gridpack, set up the environment and run
tar xf *tgz
rm -rvf *tgz
ENVNAME="vbfmlenv"
python -m venv ${ENVNAME}
source ${ENVNAME}/bin/activate
python -m pip install -e vbfml --no-cache-dir
export PYTHONPATH="${PWD}/${ENVNAME}/lib/python3.9/site-packages":${PYTHONPATH}

TARGET_DIR="vbfml/vbfml/scripts"
echo "Switching to ${TARGET_DIR}"
cd ${TARGET_DIR}

echo "Directory content---"
ls -lah .
echo "===================="

echo "Setup done: $(date)"
echo "Executing: ./train.py ${ARGS[@]}"
time ./train.py ${ARGS[@]}

# Move the trained model directory back to the top directory
# so that it gets transferred

# mv output/model_* ${JOB_DIR}
find -name 'model_v*' -type d | xargs -I {} mv {} ${JOB_DIR}
cd ${JOB_DIR}
echo "Directory content---"
ls -lah .
echo "===================="
echo "Run done: $(date)"