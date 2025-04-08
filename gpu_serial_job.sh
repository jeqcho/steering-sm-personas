#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G               # memory per node
#SBATCH --time=00:03:00

nvidia-smi                        # you can use 'nvidia-smi' for a test
