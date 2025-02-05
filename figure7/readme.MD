# Introduction

This directory is to reproduce Figure 7 in the paper.

# Run Benchmark

To the run the benchmark and get the figure all in one, run the following in shell from the current directory:

```shell
bash run.sh [rank of current machine]
```

We pass the master (rank 0) ip address to each worker machine and each worker machine actively establishes TCP connection with the master machine. Therefore, we need to run the above command manually on each machine. For example, to reproduce the results with 16 GPUs on 4 machines as in Figure 7, you should open a shell terminal on each machine and type `bash run.sh 0`, `bash run.sh 1`, `bash run.sh 2` and `bash run.sh 3` seperately in each terminal.

This will automatically run the expriments under the multi-machine setting as is described in the paper. After the benchmark completes, The generated figures (`multi_machines.pdf`) will appear in the `outputs/` directory of the rank 0 machine. Example figures can be found at `{APT_HOME}/fig_examples/figure7`.
