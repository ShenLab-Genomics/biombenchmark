#!bash

## RNABERT
echo "RNABERT"
python seq_cls.py --method RNABERT --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' --model_config 'model/RNABERT/RNABERT.json' \
    --dataset dataset/seq_cls_data --num_train_epochs 30

## RNAMSM
echo "RNAMSM"
python seq_cls.py --method RNAMSM --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' --model_config 'model/configs/RNAMSM.json' \
    --dataset dataset/seq_cls_data --num_train_epochs 30

## RNAFM
echo "RNAFM"
python seq_cls.py --method RNAFM --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --dataset dataset/seq_cls_data --num_train_epochs 30

## DNABERT
echo "DNABERT"
python seq_cls.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --dataset dataset/seq_cls_data --num_train_epochs 30 \
    --use_kmer 3

## SpliceBERT
echo "SpliceBERT"
python seq_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --dataset dataset/seq_cls_data --num_train_epochs 30

# RNAErnie
echo "RNAErnie"
python seq_cls.py --method RNAErnie \
    --model_path 'model/pretrained/RNAErnie' \
    --dataset dataset/seq_cls_data --num_train_epochs 30 \
    --logging_steps 500 \
    --use_kmer 0