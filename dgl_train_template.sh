for dataset in ogbn-products ogbn-papers100M
do
    for fanout in 10,10,10 15,10,5
    do
        for world_size in 4 8
        do
        #python examples/dgl_example.py --fan_out $fanout
        python examples/dgl_ddp.py --dataset $dataset --fan_out $fanout --world_size $world_size --logs_dir "./logs/dgl_ddp_313.csv" --tag ${dataset}_${fanout}_${world_size}
        done
    done
done


python train.py