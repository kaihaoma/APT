# Introduction

This branch contains artifacts for evaluating APT. To reproduce results in the paper, please pull the latest version from this branch. It includes codes to reproduce Figures 6, 8 and 9 in the paper:

- **Figure 6**: Epoch time for training GraphSAGE with 8 GPUs varying hidden dim, fanout and GPU cache size.
- **Figure 8**: Epoch time for training GAT with 8 GPUs varying hidden dim.
- **Figure 9**: Epoch time for training GraphSAGE with 8 GPUs using random graph partitions.

**Notes:**

1. We dont't provide the instructions here to reproduce figure 7 since it needs 16 GPUs on 4 machines and we currently don't have adequate machines on hand. However, we will provide instructions to run APT in multi-machine environments in the open-sourced codes.

2. To replicate the same experimental results as in the paper, we recommend using our provided AWS `g4dn.metal` instance with 8 NVIDIA T4 GPUs, 96 vCPUs, and 378GB memory. Running all scripts may take approximately 12 hours.

3. If you choose to use your own hardware to run the benchmarks, please ensure your hardware has adequate memory to host the large graphs, for example at least 378GB memory.

4. If you are using our provided instance, all datasets and environments are preprocessed and ready, and you can directly proceed to the `Execution` step. If not, please follow the instructions below for a proper setup.

For login instructions to our provided instance, please refer to comments `A2` and `A3` in ppopp25ae website.

# Dataset Preparation

__If you are using our provided instance, all datasets are preprocessed and ready, you can skip this step.__

This artifact requires the following datasets:

1. Ogbn-papers100M
2. Friendster
3. IGB260M

The `Ogbn-papers100M` graph is available at [OGB](https://ogb.stanford.edu/), the `Friendster` graph can be downloaded at [SNAP](https://snap.stanford.edu/data/), and the `IGB260M` graph can be obtained from [IGB](https://github.com/IllinoisGraphBenchmark/IGB-Datasets/tree/main).

After obtaining the raw dataset, save it in dgl format and use the provided `scripts/preprocess_dataset.py` to preprocess and partition the raw graph.

# Directory Structure

The repository is organized as follows:

```shell
APT
├── readme.MD # instructions
├── src # implementation of unified execution engine
├── python # python library
│   ├── npc/
│   │   ├── cost_model/ # cost model estimator module
│   │   ├── model/      # APT-adapted model
│   │   ├── utils/
│   │   ├── adaptor.py  # model adaptor module 
│   │   ├── ops.py      # python wrapper for C++ operators
│   │   ├── sampler.py  # APT-adapted sampler
│   └── setup.py
├── third_party
├── CMakeLists.txt
├── LICENSE
├── fig_examples # example output figures
├── examples  # e2e training demo with APT
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
git clone --recurse-submodules -b artifact_evaluation https://github.com/kaihaoma/APT.git
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

To execute Figure 6, 8 and 9 together, run the following command (from the root directory of this repo):
```shell
bash clean.sh
bash run.sh
```

Results will be generated in the subdirectories. See readme.MD in each subdirectory for the generated figures.

## 4.1 Build and generate each figure seperately

To only build and generate results for a specific figure, go to the directory of each figure (e.g., `cd figure6` for reproducing figure6) and run `bash run.sh`.

# License

This repo is under MIT License, see `LICENSE` for detail.