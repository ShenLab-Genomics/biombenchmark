#!/bin/bash
#SBATCH --job-name=biom-mrl-freeze
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mrl_freeze.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mrl_freeze.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "mrl-1e-3-freeze"
bash scripts/mrl/mrl_run_batch.sh 1e-3 1

echo ""
echo "mrl-1e-4-freeze"
bash scripts/mrl/mrl_run_batch.sh 1e-4 1

# echo ""
# echo "mrl-1e-5-freeze"
# bash scripts/mrl/mrl_run_batch.sh 1e-5 1