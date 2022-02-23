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

# Extract the repo gridpack, set up the environment and run
tar xf *tgz
rm -rvf *tgz
ENVNAME="vbfmlenv"
python -m venv ${ENVNAME}
source ${ENVNAME}/bin/activate
python -m pip install -e vbfml --no-cache-dir
export PYTHONPATH="${PWD}/${ENVNAME}/lib/python3.9/site-packages":${PYTHONPATH}

echo "Directory content---"
ls -lah .
echo "===================="

echo "Setup done: $(date)"
time ./scripts/train.py ${ARGS[@]}
echo "Run done: $(date)"

echo "Cleaning up."
rm -vf *.root
rm -vf ${FLIST}
echo "End: $(date)"
