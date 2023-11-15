master_addr=172.31.31.32
configs_name_all=(papers_w32_metis friendster_w32_metis igbfull_w32_metis)
# configs_name_all=(products_w32_metis)
# configs=papers_w8_metis
# configs=products_w8
logs_dir=./logs/ap/new_machines2.csv


for configs in ${configs_name_all[@]}
do
  for nproc_per_node in 8
  do
  world_size=$(($nproc_per_node*4))
  configs_path=./npc_dataset/${configs}/configs.json
  # cache_mode=dryrun
  cache_mode=none

  python examples/mp_runner.py --world_size ${world_size} --nproc_per_node=${nproc_per_node} --node_rank=$1  --master_addr=${master_addr} --master_port=12345 --logs_dir ${logs_dir} --configs_path ${configs_path} --cache_mode ${cache_mode} 
  done
done
