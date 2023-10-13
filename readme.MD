# Auto Parallel

## requirement
CUDA 11.8, DGL 1.1.2,  Pytorch 2.0.1, 

## Reuse my conda virtual python enviroment

`source .bashrc`
`conda activate ap2`

## Train on single machine
`bash ./train_configs.sh`

## Train on multi-machines
`bash ./exp_multinodes.sh 0 # on machine 0`
`bash ./exp_multinodes.sh 1 # on machine 0`