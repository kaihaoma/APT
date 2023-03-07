
for fanout in 25,10 15,10,5
do
    for cache_ratio in $(seq 0.1 0.1 1.0)
    do
    python examples/train.py --cache_ratio ${cache_ratio} --fan_out $fanout --logs_dir ./logs/npc_time_313.csv --tag 0314_try1
    done
done

python train.py