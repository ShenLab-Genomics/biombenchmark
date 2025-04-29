#!/bin/bash
#SBATCH --job-name=biom-ncRNA-mix0.1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mix_nofreeze.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mix_nofreeze.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "mix_0.1_1e-4_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-4 mix_0.1 16

echo "mix_0.1_1e-5_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-5 mix_0.1 16