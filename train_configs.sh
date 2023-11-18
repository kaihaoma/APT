
fanout_array=(10,10,10)
system_array=(NPC)
logs_dir=./logs/ap/Nov17-single-machine-SP.csv

configs_name_all=(papers_w8_metis friendster_w8_metis igbfull_w8_metis)
# configs_name=(friendster_w8_metis igbfull_w8_metis papers_w8_metis)
# configs_name=(papers_w8_metis friendster_w8_metis igbfull_w8_metis)


for configs in ${configs_name_all[@]}
do
  configs_path=./npc_dataset/${configs}/configs.json
  cache_mode=dryrun
  # cache_mode=none
  python examples/mp_runner.py --num_epochs 10 --fan_out ${fanout} --logs_dir ${logs_dir} --tag ${configs} --configs_path ${configs_path} --cache_mode ${cache_mode}
  done
done
