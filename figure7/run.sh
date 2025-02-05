rm -rf outputs
mkdir outputs

master_addr=172.31.31.32
configs_name_all=(papers_w16_metis igbfull_w16_metis friendster_w16_metis)

for configs in ${configs_name_all[@]}
do
    for nproc_per_node in 4
    do
    world_size=$(($nproc_per_node*4))

    python mp_runner.py --num_epochs 10 --fan_out 10,10,10 --world_size ${world_size} --nproc_per_node=${nproc_per_node} --node_rank=$1  --master_addr=${master_addr} --master_port=12345 --logs_dir outputs/results.csv --costmodel_log outputs/costmodel.csv --configs_path ${configs_path} --cache_mode dryrun
    done
done

if [ "$1" -eq 0 ]; then
python plot.py
fi
