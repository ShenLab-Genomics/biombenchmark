#!/bin/bash
#SBATCH --job-name=Test-biom-make-all
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j.err
#SBATCH --partition=cu

cd /public/home/shenninggroup/yny/code/biombenchmark

bash scripts/makedata_splice.sh 
