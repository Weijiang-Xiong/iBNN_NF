#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=hello.out

srun echo "Hello $USER! You are on node $HOSTNAME"

module load anaconda

python pytorch_mnist.py