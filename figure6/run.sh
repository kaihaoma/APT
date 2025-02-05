rm -rf outputs
mkdir outputs

bash run_hidden_dim.sh

bash run_fanout.sh

bash run_cache_mem.sh

python plot.py
