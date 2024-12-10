#!/bin/bash
#SBATCH --job-name=Test-biom-cls-2
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_nRC_2.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_nRC_2.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

bash scripts/cls/seq_cls_nRC_1e-5.sh