#!/bin/bash
#SBATCH --job-name=biom-ncRNA-mix0.1-pass2
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mix_nofreeze_pass2.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_mix_nofreeze_pass2.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "pass 1"
pass=1
seed=54643
echo "mix_0.01_1e-4_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-4 mix_0.01 16 ${pass} ${seed}

echo "mix_0.01_1e-5_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-5 mix_0.01 16 ${pass} ${seed}

#

echo "mix_0.1_1e-4_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-4 mix_0.1 16 ${pass} ${seed}

echo "mix_0.1_1e-5_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-5 mix_0.1 16 ${pass} ${seed}

#

echo "mix_0.5_1e-4_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-4 mix_0.5 16 ${pass} ${seed}

echo "mix_0.5_1e-5_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-5 mix_0.5 16 ${pass} ${seed}

#

echo "mix_1.0_1e-4_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-4 mix_1.0 16 ${pass} ${seed}

echo "mix_1.0_1e-5_nofreeze"
bash scripts/cls/seq_cls_batch.sh 1e-5 mix_1.0 16 ${pass} ${seed}