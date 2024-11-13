fanout_array=(10,10,10)
system_array=(NPC)
logs_dir=./zzz_ppopp2/gsplit_4gpu.csv

# configs_name_all=(friendster_w4_metis)
# configs_name_all=(products_w8)
# configs_name_all=(papers_w4_metis)
configs_name_all=(papers_w8_metis friendster_w8_metis igbfull_w8_metis)
# configs_name_all=(igbfull_w8_random)
# configs_name_all=(igbfull_w4_metis papers_w4_random)

for fanout in ${fanout_array[@]}
do
  for configs in ${configs_name_all[@]}
  do
  configs_path=./npc_dataset/${configs}/configs.json
  cache_mode=dryrun
  python examples/mp_runner.py --num_epochs 10 --fan_out ${fanout} --logs_dir ${logs_dir} --tag $random_{configs} --configs_path ${configs_path} --cache_mode ${cache_mode}
  done
done
