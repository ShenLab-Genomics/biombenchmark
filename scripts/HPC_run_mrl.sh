#!/bin/bash
#SBATCH --job-name=Test-biom-mrl
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "mrl"
bash scripts/run_mrl_test.sh
