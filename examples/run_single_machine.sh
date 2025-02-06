mkdir -p outputs

configs=papers_w8_metis

configs_path=$APT_HOME/npc_dataset/${configs}/configs.json
cache_mem=$((4*1024*1024*1024))

# Train a 3-layer GraphSAGE model with hidden dimension 32 and 4GiB GPU cache memory on 8 GPUs of a single machine
python mp_runner_hidden_dim.py --num_epochs 10 --fan_out 10,10,10 --model SAGE --cache_memory ${cache_mem} --num_hidden 32 --logs_dir outputs/results_single_machine.csv --costmodel_log outputs/costmodel_single_machine.csv --tag ${configs} --configs_path ${configs_path} --cache_mode dryrun
