mkdir -p outputs
rm -rf outputs/results_hidden_dim.csv
rm -rf outputs/costmodel_hidden_dim.csv
rm -rf outputs/hidden_dim

configs_name_all=(friendster_w8_metis)

for configs in ${configs_name_all[@]}
do
configs_path=$APT_HOME/npc_dataset/${configs}/configs.json
python mp_runner_hidden_dim.py --num_epochs 10 --fan_out 10,10,10 --logs_dir outputs/results_hidden_dim.csv --costmodel_log outputs/costmodel_hidden_dim.csv --tag ${configs} --configs_path ${configs_path} --cache_mode dryrun
done
