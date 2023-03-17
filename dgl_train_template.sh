 for fanout in 25,10 15,10,5
    do
    #python examples/dgl_example.py --fan_out $fanout
    python examples/dgl_ddp.py --fan_out $fanout --logs_dir "./logs/dgl_ddp_313.csv"
    done

python train.py