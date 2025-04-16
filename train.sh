#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Activate virtual environment
source ~/steering-sm-personas/.venv/bin/activate

# Run the training script
python train.py 