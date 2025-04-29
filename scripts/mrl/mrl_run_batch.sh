#!bash

lr_rate=$1
freeze=$2

dataset=dataset/mrl_data/mpra_data_varlen.csv
output_dir=logs/mrl_${lr_rate}_freeze${freeze}

echo "Learning rate set to: $lr_rate"
echo "Freeze base: $freeze"

common_args=(
    --dataset ${dataset}
    --output_dir ${output_dir}
    --lr ${lr_rate}
    --num_train_epochs 30
    --batch_size 64
    --logging_steps 512
    --freeze_base ${freeze}
)

## RNAFM
echo "RNAFM"
python mrl_pred.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --use_kmer 1 \
    --pad_token_id 1 \
    "${common_args[@]}"

## RNAErnie
echo "RNAErnie"
python mrl_pred.py --method RNAErnie \
    --model_path 'model/pretrained/rnaernie' \
    --use_kmer 0 \
    --pad_token_id 1 \
    "${common_args[@]}"

## ResNet
echo "ResNet"
python mrl_pred.py --method ResNet \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_config 'model/configs/PureResNet.json' \
    --use_kmer 1 \
    "${common_args[@]}"

## RNAMSM
echo "RNAMSM"
python mrl_pred.py --method RNAMSM \
    --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
    --model_config 'model/configs/RNAMSM.json' \
    --use_kmer 1 \
    "${common_args[@]}"

## RNABERT
echo "RNABERT"
python mrl_pred.py --method RNABERT \
    --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' \
    --model_config 'model/RNABERT/RNABERT.json' \
    --use_kmer 1 \
    "${common_args[@]}"

## DNABERT
echo "DNABERT"
python mrl_pred.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --use_kmer 3 \
    "${common_args[@]}"

## DNABERT2
echo "DNABERT2"
python mrl_pred.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --pad_token_id 3 \
    --use_kmer 0 \
    "${common_args[@]}"

## SpliceBERT
echo "SpliceBERT"
python mrl_pred.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --use_kmer 1 \
    "${common_args[@]}"

# NucleotideTransformer
python mrl_pred.py --method NucleotideTransformer \
    --model_path 'model/pretrained/NucleotideTransformer2' \
    --use_kmer 0 \
    "${common_args[@]}"

# GENA-LM
python mrl_pred.py --method GENA-LM-base \
    --model_path 'model/pretrained/GENA-LM/gena-lm-bert-base-t2t' \
    --use_kmer 0 \
    --pad_token_id 3 \
    "${common_args[@]}"

# GENA-LM
python mrl_pred.py --method GENA-LM-large \
    --model_path 'model/pretrained/GENA-LM/gena-lm-bert-large-t2t' \
    --use_kmer 0 \
    --pad_token_id 3 \
    "${common_args[@]}"

## UTRLM
echo "UTRLM"
python mrl_pred.py --method UTRLM \
    --vocab_path 'model/vocabs/UTRLM.txt' \
    --model_path 'model/UTRlm/model.pt' \
    --use_kmer 1 \
    "${common_args[@]}"

## Optimus
echo "Optimus"
python mrl_pred.py --method Optimus \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --use_kmer 1 \
    "${common_args[@]}"