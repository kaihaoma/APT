mkdir -p outputs
rm -rf outputs/results_fanout.csv

configs_name_all=(papers_w8_metis friendster_w8_metis igbfull_w8_metis)
fanout_array=(5,10 10,15 10,10,10 10,15,20)

for configs in ${configs_name_all[@]}
do
    for fanout in ${fanout_array[@]}
    do
    configs_path=$APT_HOME/npc_dataset/${configs}/configs.json
    python mp_runner_fanout.py --num_epochs 10 --fan_out ${fanout} --logs_dir outputs/results_fanout.csv --tag ${configs} --configs_path ${configs_path} --cache_mode dryrun
    done
done
