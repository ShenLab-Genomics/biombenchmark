#!/bin/bash
#SBATCH --job-name=Test-biom-cls-1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_nRC_1.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_nRC_1.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

bash scripts/cls/seq_cls_nRC_1e-4.sh