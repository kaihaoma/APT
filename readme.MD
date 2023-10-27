# Auto Parallel

## requirement
CUDA 11.8, DGL 1.1.2,  Pytorch 2.0.1, 
`conda create -n ap python=3.9 #create conda virtual enviroment python 3.9`
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia #conda install pytorch-gpu CUDA 11.8`
`conda install -c dglteam/label/cu118 dgl #conda install dgl CUDA 11.8`

## Reuse my conda virtual python enviroment

`source .bashrc`
`conda activate ap2`

## Train on single machine
`bash ./train_configs.sh`

## Train on multi-machines
`bash ./exp_multinodes.sh 0 # on machine 0`
`bash ./exp_multinodes.sh 1 # on machine 0`