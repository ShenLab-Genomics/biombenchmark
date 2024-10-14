#!/bin/bash
#SBATCH --job-name=Test-biom-m6-101
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6a101.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6a101.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

# bash scripts/run_m6a_test.sh
# echo "m6A-101bp-miclip"
bash scripts/run_m6a_test_m6A_101.sh

# bash run_m6a_append_DNABERT2.sh
