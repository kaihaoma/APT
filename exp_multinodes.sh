master_addr=172.31.40.2
fanout_array=(10,10,10)
configs=papers_w8_metis
#configs=products_w8
logs_dir=./logs/ap/new_machines2.csv

for num_localnode_feats_in_workers in 0
do
  for fanout in ${fanout_array[@]}
  do
    for nproc_per_node in 4
    do
    world_size=$(($nproc_per_node*2))
    configs_path=./npc_dataset/${configs}/configs.json
    cache_mode=dryrun
    
    python examples/mp_runner.py --world_size ${world_size} --nproc_per_node=${nproc_per_node} --node_rank=$1  --master_addr=${master_addr} --master_port=12345 --num_localnode_feats_in_workers ${num_localnode_feats_in_workers} --logs_dir ${logs_dir} --tag multinodes_Oct30 --configs_path ${configs_path} --cache_mode ${cache_mode} 

    done
    #sleep 5
  done
done
