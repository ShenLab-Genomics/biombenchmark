#!/bin/bash
#SBATCH --job-name=Test-biom-splice-5cls-part1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_sp15.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_sp15.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

# bash scripts/run_splice_train_test.sh
bash scripts/run_splice_train_test_15.sh
