rm -rf outputs
mkdir outputs

configs_name_all=(papers_w8_random friendster_w8_random igbfull_w8_random)

for configs in ${configs_name_all[@]}
do
configs_path=$APT_HOME/npc_dataset/${configs}/configs.json
python mp_runner_random_partition.py --num_epochs 10 --fan_out 10,10,10 --logs_dir outputs/results.csv --tag ${configs} --configs_path ${configs_path} --cache_mode dryrun
done

python plot.py
