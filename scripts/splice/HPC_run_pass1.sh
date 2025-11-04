#!/bin/bash
#SBATCH --job-name=biom-sp-3-pass1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_sp3_pass1.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_sp3_pass1.err
#SBATCH --partition=gpu03
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "pass 1"
pass=1
seed=54643

echo "splice-1e-4-3class"
bash scripts/splice/splice_normal.sh 1e-4 ${pass} ${seed}

echo "splice-1e-5-3class"
bash scripts/splice/splice_normal.sh 1e-5 ${pass} ${seed}
