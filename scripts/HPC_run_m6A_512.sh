#!/bin/bash
#SBATCH --job-name=Test-biom-m6-512
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6A512.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6A512.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

# bash scripts/run_m6a_test.sh
bash scripts/run_m6a_test_m6A_512.sh
