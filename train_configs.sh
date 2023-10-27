
fanout_array=(10,10,10)
system_array=(NPC)
logs_dir=./logs/ap/new_machines.csv

configs_name_all=(papers_w8_metis friendster_w8_metis igblarge_w8_metis)
configs_name=(products_w8)


for fanout in ${fanout_array[@]}
do
  for configs in ${configs_name[@]}
  do
  configs_path=./npc_dataset/${configs}/configs.json
  python examples/mp_runner.py  --fan_out ${fanout} --logs_dir ${logs_dir} --tag Oct10_${configs} --configs_path ${configs_path}
  done
done
