#!/bin/bash
#SBATCH --job-name=app-biom-cls
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_nRC_ap1.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_nRC_ap1.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu01
#SBATCH --gres=gpu:1

echo "NucleotideTransformer - nrc 1e-4"
python seq_cls.py --method NucleotideTransformer \
    --model_path 'model/pretrained/NucleotideTransformer2' \
    --output_dir 'logs/nRC_1e-4' \
    --dataset dataset/seq_cls_data \
    --batch_size 40 \
    --num_train_epochs 30 \
    --use_kmer 0 \
    --lr 1e-4


# echo "NucleotideTransformer - nrc 5e-4"
# python seq_cls.py --method NucleotideTransformer \
#     --model_path 'model/pretrained/NucleotideTransformer2' \
#     --output_dir 'logs/nRC_5e-4' \
#     --dataset dataset/seq_cls_data \
#     --batch_size 40 \
#     --num_train_epochs 30 \
#     --use_kmer 0 \
#     --lr 5e-4

echo "NucleotideTransformer - nrc 1e-5"
python seq_cls.py --method NucleotideTransformer \
    --model_path 'model/pretrained/NucleotideTransformer2' \
    --output_dir 'logs/nRC_1e-5' \
    --dataset dataset/seq_cls_data \
    --batch_size 40 \
    --num_train_epochs 30 \
    --use_kmer 0 \
    --lr 1e-5