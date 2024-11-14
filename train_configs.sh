fanout_array=(10,10,10)
# For exp of varying fanout
# fanout_array=(5,10, 10,15 10,10,10 10,15,20)
system_array=(NPC)
# log file, change it to your own directory
logs_dir=./zzz_ppopp2/gsplit_4gpu.csv


configs_name_all=(papers_w8_metis friendster_w8_metis igbfull_w8_metis)
# For exp of random partiton
#configs_name_all=(papers_w8_random friendster_w8_random igbfull_w8_random)


for fanout in ${fanout_array[@]}
do
  for configs in ${configs_name_all[@]}
  do
  configs_path=./npc_dataset/${configs}/configs.json
  cache_mode=dryrun
  python examples/mp_runner.py --num_epochs 10 --fan_out ${fanout} --logs_dir ${logs_dir} --tag $random_{configs} --configs_path ${configs_path} --cache_mode ${cache_mode}
  done
done
