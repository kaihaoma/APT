mkdir -p outputs

master_addr=172.31.31.32
configs=papers_w16_metis
nproc_per_node=4
world_size=$(($nproc_per_node*4))
cache_mem=$((4*1024*1024*1024))

# Train a 3-layer GraphSAGE model with hidden dimension 32 and 4GiB GPU cache memory on 4 machine each with 4 GPUs
python mp_runner.py --num_epochs 10 --fan_out 10,10,10 --model SAGE --cache_memory ${cache_mem} --num_hidden 32 --world_size ${world_size} --nproc_per_node=${nproc_per_node} --node_rank=$1  --master_addr=${master_addr} --master_port=12345 --logs_dir outputs/results_multi_machine.csv --costmodel_log outputs/costmodel_multi_machine.csv --configs_path ${configs_path} --cache_mode dryrun
