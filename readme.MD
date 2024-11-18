# Introduction

This branch contains artifacts for evaluating APT. To reproduce results in the paper, please pull the latest version from this branch. It includes codes to reproduce Figures 6, 7, 8 and 9 from the paper:

- **Figure 6**: Epoch time for training GraphSAGE with 8 GPUs on a single machine varying hidden dim, fanout and GPU cache size.
- **Figure 7**: Epoch time for training GraphSAGE with 16 GPUs on 4 machines varying hidden dim.
- **Figure 8**: Epoch time for training GAT with 8 GPUs on a single machine varying hidden dim.
- **Figure 9**: Epoch time for training GraphSAGE with 8 GPUs on a single machine using random graph partitions.


To replicate the same experimental results as in the paper, we recommend using our provided AWS `g4dn.metal` instance with NVIDIA T4 GPUs, 96 vCPUs, and 378GB memory. Running all scripts may take approximately 12 hours.

If you leverage your own hardware to reprodce the results, please ensure your hardware has adequate memory to host the large graphs, for example at least 378GB memory.

If you are using our provided instance, all datasets and environments are preprocessed and ready, and you can directly proceed to the `Execution` step. If not, please follow the instructions below for a proper setup.

For login instructions to our provided instance, please refer to comments `A2` and `A3` in ppopp25ae website.

# Dataset Preparation

__If you are using our provided instance, all datasets are preprocessed and ready, you can skip this step.__

This artifact requires the following datasets:

1. Ogbn-papers100M
2. Friendster
3. IGB260M

The `Ogbn-papers100M` graph is available at [OGB](https://ogb.stanford.edu/), the `Friendster` graph can be downloaded at [SNAP](https://snap.stanford.edu/data/), and the `IGB260M` graph can be obtained from [IGB](https://github.com/IllinoisGraphBenchmark/IGB-Datasets/tree/main).

After obtaining the dataset, use the provided scripts to preprocess and partition the raw graph (from the root directory of this repo):
```shell
python scripts/preprocess_graph.py --dataset xxx --path xxx
python scripts/preprocess_graph.py --dataset xxx --path xxx
python scripts/preprocess_graph.py --dataset xxx --path xxx
```

# Directory Structure

The repository contains four directories as follows:

```shell
gsampler-artifact-evaluation
├── readme.MD
├── fig_examples # Example output figures.
├── examples  # E2E training demo with APT
├── figure6   # reproduce figure6
├── figure7   # reproduce figure7
├── figure8   # reproduce figure8
├── figure9   # reproduce figure9
├── clean.sh  # clean all results
└── run.sh    # run all experiments
```

# Setup

__If you are using our provided instance, run `conda activet ap2` to use the pre-configured conda environment and skip this step.__

Follow these instructions to prepare and install APT with all required dependcies:

1. Git clone the repo:
```shell
xxx
```

## 3.1 Build APT

From the root directory of this repo:
```shell
mkdir build; cd build
cmake ..; make -j20
```

## 3.2 Install APT

From the root directory of this repo:
```shell
cd python; python setup.py install
```

# Execution

To execute Figure 6, 7, 8 and 10 together, run the following command (from the root directory of this repo):

```
bash clean.sh
bash run.sh
```

Results will be generated in the subdirectories. See readme.MD in each subdirectory for the generated figures.

## 4.1 Build and generate each figure seperately

To only build and generate results for a specific figure, go to the directory of each figure (e.g., `cd figure6` for reproducing figure6) and run `bash run.sh`.
