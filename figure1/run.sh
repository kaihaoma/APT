rm -rf outputs
mkdir outputs

bash run_input_dim.sh

bash run_hidden_dim.sh

python plot.py
