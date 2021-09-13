# ML training tools for VBF H(inv)

[![Coverage Status](https://coveralls.io/repos/github/AndreasAlbert/vbfml/badge.svg?branch=main)](https://coveralls.io/github/AndreasAlbert/vbfml?branch=main)


## Features

| Feature | Status |
| ------- | ------ |
| Single-file reading | done |
| Multi-file reading | done |
| Keras conformant output | done |
| Train / validation / test splitting | done |
| Per-event weights from input tree | done |
| Per-dataset weight modifiers (xs, sumw) | done |


## Setup

Note: Set up a [python virtual environment](https://docs.python.org/3/tutorial/venv.html) before installing to make your life easier.

```
git clone git@github.com:AndreasAlbert/vbfml.git
python3 -m pip install -e vbfml
```

## Contributing

Code quality is ensured through extensive testing. When developing a new feature, please write unit tests at the same time. Check out the tests directory to see existing tests for inspiration. 
Tests are executed using pytest, which is also automatically done for each pull request through github actions. Make sure that all tests pass:

```bash
cd vbfml
python3 -m pytest
```

To avoid having to deal with coding style questions, all code is formatted with [black](https://github.com/psf/black). Please format your code before committing:

```bash
# individual file:
black myfile.py

# or whole folder:
black vbfml
```

## Usage example

Check out the scripts/train.py script for an example training work flow. The scripts/analyze_training.py script can be used to create typical postprocessing plots, such as variable distributions. It can be extended with further postprocessing:

```bash
./scripts/analyze_training.py plot /path/to/training/folder/
```