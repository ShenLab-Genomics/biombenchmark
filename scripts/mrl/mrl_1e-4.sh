#!bash

## RNAFM
echo "RNAFM"
python mrl_pred.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --use_kmer 1

echo "RNAErnie"
python mrl_pred.py --method RNAErnie \
    --model_path 'model/pretrained/rnaernie' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --use_kmer 0

echo "ResNet"
python mrl_pred.py --method ResNet \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_config 'model/configs/PureResNet.json' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --use_kmer 1

## RNAMSM
echo "RNAMSM"
python mrl_pred.py --method RNAMSM \
    --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
    --model_config 'model/configs/RNAMSM.json' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --use_kmer 1

## RNABERT
echo "RNABERT"
python mrl_pred.py --method RNABERT --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' \
    --model_config 'model/RNABERT/RNABERT.json' \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --batch_size 64 \
    --num_train_epochs 30 \
    --logging_steps 200 \
    --use_kmer 1


## DNABERT
echo "DNABERT"
python mrl_pred.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --batch_size 64 \
    --num_train_epochs 30 \
    --logging_steps 200 \
    --use_kmer 3

## DNABERT2
echo "DNABERT2"
python mrl_pred.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --logging_steps 100 \
    --pad_token_id 3 \
    --use_kmer 0

# SpliceBERT
echo "SpliceBERT"
python mrl_pred.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --logging_steps 100 \
    --use_kmer 1


echo "UTRLM"
python mrl_pred.py --method UTRLM \
    --vocab_path 'model/vocabs/UTRLM.txt' \
    --model_path 'model/UTRlm/model.pt' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --use_kmer 1

echo "Optimus"
python mrl_pred.py --method Optimus \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --use_kmer 1