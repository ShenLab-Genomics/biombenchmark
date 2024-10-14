#!bash

## RNAFM
echo "RNAFM"
python m6a_cls.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --output_dir 'logs/m6a_512_seq' \
    --num_train_epochs 10 \
    --class_num 2 \
    --batch_size 32 \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --use_kmer 1 \
    --lr 1e-4 

## RNAMSM
echo "RNAMSM"
python m6a_cls.py --method RNAMSM \
    --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
    --model_config 'model/configs/RNAMSM.json' \
    --output_dir 'logs/m6a_512_seq' \
    --num_train_epochs 10 \
    --class_num 2 \
    --batch_size 32 \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --use_kmer 1

# RNAErnie
echo "RNAErnie"
python m6a_cls.py --method RNAErnie \
    --model_path 'model/pretrained/RNAErnie' \
    --output_dir 'logs/m6a_512_seq' \
    --num_train_epochs 10 \
    --class_num 2 \
    --batch_size 32 \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --use_kmer 0 

## RNABERT
echo "RNABERT"
python m6a_cls.py --method RNABERT --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' \
    --model_config 'model/RNABERT/RNABERT.json' \
    --output_dir 'logs/m6a_512_seq' \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --batch_size 64 \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --use_kmer 1

echo "RNABERT_RAW"
python m6a_cls.py --method RNABERT_RAW --vocab_path 'model/RNABERT/vocab.txt' \
    --model_config 'model/RNABERT/RNABERT.json' \
    --output_dir 'logs/m6a_512_seq' \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --batch_size 64 \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --use_kmer 1

## DNABERT
echo "DNABERT"
python m6a_cls.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --output_dir 'logs/m6a_512_seq' \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --batch_size 32 \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --use_kmer 3

## DNABERT2
echo "DNABERT2"
python m6a_cls.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --output_dir 'logs/m6a_512_seq' \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --num_train_epochs 10 \
    --batch_size 32 \
    --class_num 2 \
    --logging_steps 50 \
    --use_kmer 0

# SpliceBERT
echo "SpliceBERT"
python m6a_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --output_dir 'logs/m6a_512_seq' \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --batch_size 32 \
    --num_train_epochs 10 \
    --logging_steps 50 


