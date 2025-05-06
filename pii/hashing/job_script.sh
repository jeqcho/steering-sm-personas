#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

python hash_did.py --folder_name "/home/s4yor1/scratch/pii_removed/processed_2_clusters"
python hash_did.py --folder_name "/home/s4yor1/scratch/pii_removed/processed_100_clusters"