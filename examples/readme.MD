# Single-machine Training

Script `run_single_machine.sh` contains an example for training a 3-layer GraphSAGE model with hidden dimension 32 and 4GiB GPU cache memory on 8 GPUs of a single machine. The dataset is `Ogbn-papers100M`.

Run `bash run_single_machine.sh` to play with the example. Arguments passed to the program (e.g., cache memory, hidden dim, input dim, fanout, etc.) can be adjusted to test the performance of APT under various scenarios.

Logging results will appear in the `outputs/` directory.

# Multi-machine Training

Script `run_multi_machine.sh` contains an example for training a 3-layer GraphSAGE model with hidden dimension 32 and 4GiB GPU cache memory on 4 machine each with 4 GPUs. The dataset is `Ogbn-papers100M`.

We pass the master (rank 0) ip address to each worker machine and each worker machine actively establishes TCP connection with the master machine. Therefore, we need to run the above command manually on each machine. For example, to reproduce the results with 16 GPUs on 4 machines as in Figure 7, you should open a shell terminal on each machine and type `bash run_multi_machine.sh 0`, `bash run_multi_machine.sh 1`, `bash run_multi_machine.sh 2` and `bash run_multi_machine.sh 3` seperately in each terminal.

Logging results will appear in the `outputs/` directory of the rank 0 machine.
