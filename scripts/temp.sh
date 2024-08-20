#!bash

## DNABERT
python seq_cls.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --dataset dataset/seq_cls_data --num_train_epochs 2 --use_kmer False

## SpliceBERT
python seq_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --dataset dataset/seq_cls_data --num_train_epochs 2
