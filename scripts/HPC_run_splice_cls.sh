#!/bin/bash
#SBATCH --job-name=Test-biom-splice-1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

bash scripts/run_splice_train_test.sh
