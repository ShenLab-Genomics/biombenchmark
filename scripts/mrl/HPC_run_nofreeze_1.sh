#!/bin/bash
#SBATCH --job-name=biom-mrl-nofreeze-pass1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mrl_nofreeze_pass1.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mrl_nofreeze_pass1.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "pass 1"
pass=1
seed=54643

echo "mrl-1e-3-nofreeze"
bash scripts/mrl/mrl_run_batch.sh 1e-3 0 ${pass} ${seed}

echo ""
echo "mrl-1e-4-nofreeze"
bash scripts/mrl/mrl_run_batch.sh 1e-4 0 ${pass} ${seed}