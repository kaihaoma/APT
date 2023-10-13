#bash ./kill.sh
#fanout_array=(5,10,15 10,10,10)
fanout_array=(10,10,10)

#uva_size_array=(16)
#greedy_feat_ratio_array=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0)
system_array=(NPC)
logs_dir=./logs/ap/new_machines.csv
#cache_memory_list=(1)
#papers
dataset=ogbn-papers100M
input_dim_array=(128)
#input_dim_array=(256)
num_classes=172
#num_classes=3
#papers-8gpus
#configs_path=./npc_dataset/rebla-papers_w8_0717/configs.json
#configs_name=(papers_w4_rebla_0722 papers_w8_rebla_0722)
configs_name_all=(papers_w8_metis friendster_w8_metis igblarge_w8_metis)
configs_name=(papers_w8_metis)

#configs_original=(papers_w4 papers_w8)
#papers-8gpus-balanced
#world_size=8
#idx_mem_path=./npc_dataset/rebla-papers/sorted_idx_mem.pt
#graph_path=./npc_dataset/rebla-papers/rebla-papers-pure.bin
#min_vids=0,14477529,25259156,38927312,53383192,67859936,82335215,96606010,111059956
#sampling_path=./sampling_all/npc_rebla/rebla-papers

#papers-4gpus
#world_size=4
#min_vids=0,28842663,54987214,82217330,111059956
#graph_path=./npc_dataset/ogbn-papers100M4_pure.bin
#sampling_path=./sampling_all/npc2/papersM4
#products

#10 10 10





for fanout in ${fanout_array[@]}
do
  for input_dim in ${input_dim_array[@]}
  do
      for configs in ${configs_name[@]}
      do
      configs_path=./npc_dataset/${configs}/configs.json
      cache_mode=dryrun
      python examples/mp_runner.py  --input_dim ${input_dim} --num_epochs 10 --cache_memory 1 --fan_out ${fanout} --logs_dir ${logs_dir} --tag Oct10_${configs} --cache_mode ${cache_mode} --configs_path ${configs_path}
      done
  done
done

echo "[Note]Done train_configs.sh, run hang-on scripts" 
python train.py