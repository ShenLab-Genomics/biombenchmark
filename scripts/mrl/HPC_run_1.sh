#!/bin/bash
#SBATCH --job-name=biom-mrl-1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mrl_1-4.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mrl_1-4.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "mrl-1e-4"
bash scripts/mrl/mrl_1e-4.sh
