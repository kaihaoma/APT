master_addr=172.31.31.32
fanout_array=(10,10,10)
configs=papers_w16_metis
#configs=products_w8
logs_dir=./logs/acc_multimachine.csv

for num_localnode_feats_in_workers in 0
do
  for fanout in ${fanout_array[@]}
  do
    for nproc_per_node in 4
    do
    world_size=$(($nproc_per_node*4))
    configs_path=./npc_dataset_acc2/${configs}/configs.json
    cache_mode=dryrun
    
    python examples/mp_runner.py --num_epochs 50 --fan_out ${fanout} --world_size ${world_size} --nproc_per_node=${nproc_per_node} --node_rank=$1  --master_addr=${master_addr} --master_port=12345 --logs_dir ${logs_dir} --configs_path ${configs_path} --cache_mode ${cache_mode} --dataset ogbn-papers100M --input_dim 128 --debug

    done
    #sleep 5
  done
done
