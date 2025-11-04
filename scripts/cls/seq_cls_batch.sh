#!bash

lr_rate=$1
data_group=$2
class_num=$3
pass=$4
seed=$5
echo "Learning rate set to: $lr_rate"

common_args=(
    --dataset dataset/seq_cls_data
    --data_group ${data_group}
    --output_dir logs/nRC_${lr_rate}_${data_group}_${pass}
    --num_train_epochs 30
    --batch_size 40
    --logging_steps 512
    --class_num ${class_num}
    --lr ${lr_rate}
    --seed ${seed}
)

## RNAFM
echo "RNAFM"
python seq_cls.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    "${common_args[@]}"

## RNAMSM
echo "RNAMSM"
python seq_cls.py --method RNAMSM \
    --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
    --model_config 'model/configs/RNAMSM.json' \
    --pad_token_id 1 \
    "${common_args[@]}"


# RNAErnie
echo "RNAErnie"
python seq_cls.py --method RNAErnie \
    --model_path 'model/pretrained/rnaernie' \
    --use_kmer 0 \
    "${common_args[@]}"

## RNABERT
echo "RNABERT"
python seq_cls.py --method RNABERT \
    --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' \
    --model_config 'model/RNABERT/RNABERT.json' \
    "${common_args[@]}"

## DNABERT
echo "DNABERT"
python seq_cls.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --use_kmer 3 \
    "${common_args[@]}"


## SpliceBERT
echo "SpliceBERT"
python seq_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    "${common_args[@]}"

## DNABERT2
echo "DNABERT2"
python seq_cls.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --use_kmer 0 \
    --pad_token_id 3 \
    "${common_args[@]}"


echo "NucleotideTransformer"
python seq_cls.py --method NucleotideTransformer \
    --model_path 'model/pretrained/NucleotideTransformer2' \
    --use_kmer 0 \
    "${common_args[@]}"


echo "GENA-LM-base"
python seq_cls.py --method GENA-LM-base \
    --model_path 'model/pretrained/GENA-LM/gena-lm-bert-base-t2t' \
    --use_kmer 0 \
    --pad_token_id 3 \
    "${common_args[@]}"

echo "GENA-LM-large"
python seq_cls.py --method GENA-LM-large \
    --model_path 'model/pretrained/GENA-LM/gena-lm-bert-large-t2t' \
    --use_kmer 0 \
    --pad_token_id 3 \
    "${common_args[@]}"

echo "UTRLM"
python seq_cls.py --method UTRLM \
    --vocab_path 'model/vocabs/UTRLM.txt' \
    --model_path 'model/UTRLM/model.pt' \
    --use_kmer 1 \
    --pad_token_id 0 \
    "${common_args[@]}"

echo ncRDense
python seq_cls.py --method ncRDense \
    "${common_args[@]}"