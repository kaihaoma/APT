fanout=15,10,5
#fanout=10,10,10
system=NPC
#system in NPC, DGL-global, DSP, DSP-1hop
feat_cache_ratio=0.05
graph_cache_ratio=0.05
#papers 4GPUs
graph_path=./npc_dataset/ogbn-papers100M4.bin
min_vids=0,28842663,54987214,82217330,111059956
num_classes=172
input_dim=128

#papers 8GPUs
#graph_path=./npc_dataset/ogbn-papers100MM8.bin
#min_vids=0,13717553,27592167,42068855,56543827,71017008,85487258,99909142,111059956
#num_classes=172
#input_dim=128

#default setting is 4GPUs products
#python examples/train.py 

python examples/train.py --system ${system} --feat_cache_ratio ${feat_cache_ratio} --graph_cache_ratio ${graph_cache_ratio} --graph_path ${graph_path} --min_vids ${min_vids} --num_classes ${num_classes} --input_dim ${input_dim}