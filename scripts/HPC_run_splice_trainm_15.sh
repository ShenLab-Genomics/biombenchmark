#!/bin/bash
#SBATCH --job-name=Test-biom-ma-splice-15cls
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_sp15train.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_sp15train.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

# bash scripts/run_splice_train_test.sh
# bash scripts/run_splice_train_test_15.sh


python pretrain.py --method RNAMAMBA \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --num_train_epochs 5 \
    --class_num 18 \
    --batch_size 12 \
    --output_dir 'model/fine_tuned/Splicing/RNAMAMBA_15class' \
    --dataset_train 'dataset/splice_data/gtex_500_15tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_15tis/dataset_test.h5' \
    --use_kmer 1