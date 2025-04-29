#!/bin/bash
#SBATCH --job-name=biom-sp-3-append
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_sp3.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_sp3.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "splice-1e-4-3class"
bash scripts/splice/splice_normal_append.sh 1e-4

echo "splice-1e-5-3class"
bash scripts/splice/splice_normal_append.sh 1e-5
