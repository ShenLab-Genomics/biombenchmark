#!/bin/bash
#SBATCH --job-name=biom-m6-1.0-pass1-101
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6a101_4-1.0-pass1-101.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6a101-4_1.0-pass1-101.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "pass 1"
pass=1
seed=54643

echo "m6A-101bp-miclip-1e-4-1.0"
bash scripts/m6A/m6a_cls_batch.sh 1e-4 101bp 1.0 ${pass} ${seed}

echo "m6A-101bp-miclip-1e-5-1.0"
bash scripts/m6A/m6a_cls_batch.sh 1e-5 101bp 1.0 ${pass} ${seed}