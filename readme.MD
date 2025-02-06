# APT: Adaptive Parallel Training for Graph Neural Networks

This is the repository containing the source code and artifact for the PPoPP'25 paper "APT: Adaptive Parallel Training for Graph Neural Networks". To reproduce results in the paper, please checkout to the `artifact_evaluation` branch for instructions.

# Installation

Follow these steps to prepare and install APT with all required dependcies.

## Software Prerequisites

To install and use APT, the following dependencies is required. We suggest you create a new conda environment for this.

* python >= 3.9
* cmake >= 3.27.4
* CUDA >= 11.8
* DGL >= 1.1.2
* Pytorch >= 2.0.1

## Clone APT

Git clone the repo:
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

# Datasets

We need to partition the graph and output the required format before APT can operate on it. We provide the code script to prepared the dataset in `scripts/preprocess_dataset.py`. Especially, you will need to preprare your own dataset in advance in the binary format that can be loaded with `dgl.load_graphs()`. The script goes through the following steps.

1. Load your inital graph with `dgl.load_graphs()`.
2. Partition the graph using `dgl.distributed.partition_graph()`, either with Metis or random partitioning.
3. Calculate the ID offsets of each graph partition.
4. Reorder the whole graph to make the IDs in each graph partition contiguous.
5. Store the reordered graph and configs of the partitions in the output path.
6. Count dryrun results (e.g., node hotness) if indicated.

Example config files are in `npc_dataset/`.

# License

This repo is under MIT License, see `LICENSE` for further information.
