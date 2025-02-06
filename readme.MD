# APT: Adaptive Parallel Training for Graph Neural Networks

This is the repository containing the source code and artifact for the PPoPP'25 paper "APT: Adaptive Parallel Training for Graph Neural Networks". To reproduce results in the paper, please checkout to the `artifact_evaluation` branch for instructions.

# Installation

## Software Prerequisites

To install and use APT, the following dependencies is required. We suggest you create a new conda environment for this.

* python >= 3.9
* cmake >= 3.27.4
* CUDA >= 11.8
* DGL >= 1.1.2
* Pytorch >= 2.0.1

## Clone APT

Follow these instructions to prepare and install APT with all required dependcies:

1. Git clone the repo:
```shell
git clone --recurse-submodules https://github.com/kaihaoma/APT.git
```

## Build APT

From the root directory of this repo:
```shell
mkdir build; cd build
cmake ..; make -j20
```

## Install APT

From the root directory of this repo:
```shell
cd python; python setup.py install
```

# Usage

We provide shell scripts for running both single-machine and multi-machine GNN training. See instructions in `examples/` for detail.

# License

This repo is under MIT License, see `LICENSE` for further information.
