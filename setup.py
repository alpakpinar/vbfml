from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = 'vbfml',
    version = '0.0.1',
    url = 'https://github.com/bu-cms/vbfml',
    author = 'Andreas Albert',
    author_email = 'andreas.albert@cern.ch',
    description = 'Tools for ML-based analysis for VBF H(inv)',
    packages = find_packages(),    
    install_requires = requirements,
    scripts=[],
)
