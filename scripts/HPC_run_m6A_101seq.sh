#!/bin/bash
#SBATCH --job-name=Test-biom-m6-101seq
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6Aseq101.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6Aseq101.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

# bash scripts/run_m6a_test.sh
echo "m6A-101bp-m6Aseq"
bash scripts/run_m6a_test_m6A_101seq.sh
