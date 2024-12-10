#!bash

## RNAFM
echo "RNAFM"
python seq_cls.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --output_dir 'logs/nRC_1e-4' \
    --num_train_epochs 30 \
    --dataset dataset/seq_cls_data \
    --batch_size 40 \
    --pad_token_id 1

## RNAMSM
echo "RNAMSM"
python seq_cls.py --method RNAMSM \
    --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
    --model_config 'model/configs/RNAMSM.json' \
    --output_dir 'logs/nRC_1e-4' \
    --num_train_epochs 30 \
    --dataset dataset/seq_cls_data \
    --batch_size 40 \
    --pad_token_id 1

# RNAErnie
echo "RNAErnie"
python seq_cls.py --method RNAErnie \
    --model_path 'model/pretrained/rnaernie' \
    --output_dir 'logs/nRC_1e-4' \
    --dataset dataset/seq_cls_data \
    --num_train_epochs 30 \
    --batch_size 40 \
    --use_kmer 0

## RNABERT
echo "RNABERT"
python seq_cls.py --method RNABERT \
    --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' \
    --model_config 'model/RNABERT/RNABERT.json' \
    --output_dir 'logs/nRC_1e-4' \
    --num_train_epochs 30 \
    --dataset dataset/seq_cls_data \
    --batch_size 40



## DNABERT
echo "DNABERT"
python seq_cls.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --dataset dataset/seq_cls_data \
    --output_dir 'logs/nRC_1e-4' \
    --num_train_epochs 30 \
    --batch_size 40 \
    --use_kmer 3 

## SpliceBERT
echo "SpliceBERT"
python seq_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --output_dir 'logs/nRC_1e-4' \
    --dataset dataset/seq_cls_data \
    --batch_size 40 \
    --num_train_epochs 30



## DNABERT2
echo "DNABERT2"
python seq_cls.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --dataset dataset/seq_cls_data \
    --output_dir 'logs/nRC_1e-4' \
    --num_train_epochs 30 \
    --batch_size 40 \
    --use_kmer 0 \
    --pad_token_id 3

echo "NucleotideTransformer - nrc 1e-4"
python seq_cls.py --method NucleotideTransformer \
    --model_path 'model/pretrained/NucleotideTransformer2' \
    --output_dir 'logs/nRC_1e-4' \
    --dataset dataset/seq_cls_data \
    --batch_size 40 \
    --num_train_epochs 30 \
    --use_kmer 0 \
    --lr 1e-4