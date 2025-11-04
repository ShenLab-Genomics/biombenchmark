#!/bin/bash
#SBATCH --job-name=biom-mrl-freeze-pass1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mrl_freeze_pass1.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mrl_freeze_pass1.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "pass 1"
pass=1
seed=54643

echo "mrl-1e-3-freeze"
bash scripts/mrl/mrl_run_batch.sh 1e-3 1 ${pass} ${seed}

echo ""
echo "mrl-1e-4-freeze"
bash scripts/mrl/mrl_run_batch.sh 1e-4 1 ${pass} ${seed}