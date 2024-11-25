# Introduction

This directory is to reproduce Figure 6 in the paper.

# Run Benchmark

To the run the benchmarks and get the figures all in one, run the following in shell from the current directory:

```shell
bash run.sh
```

This will automatically run the expriments for `hidden dimension`, `fanout` and `GPU cache memory` as is described in the paper. After the benchmark completes, The generated figures (`hidden_dim.pdf`, `fanout.pdf` and `gpu_cache_mem.pdf`) will appear in the `outputs/` directory. Example figures can be found at `{APT_HOME}/fig_examples/figure6`.

# Run Specific Experiments

If you only want to run a specific experiment, run one of the following:

```shell
bash run_hidden_dim.sh
bash run_fanout.sh
bash run_cache_mem.sh
```

## Plot Outputs

To generate figures as in the paper (if not using `bash run.sh`), run the following command in shell:

```shell
python plot.py
```

All three figures will be produced in the `outputs/` directory.