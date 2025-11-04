#!bash

lr_rate=$1
length=$2
split=$3
pass=$4
seed=$5

traindata=dataset/m6a_data/miCLIP/${length}/${split}_train.fa
testdata=dataset/m6a_data/miCLIP/${length}/test.fa
output_dir=logs/m6a_${length}_clip_${lr_rate}_${split}_${pass}_${seed}

echo "Learning rate set to: $lr_rate"
echo "Split set to: $split"
echo "Sequence length set to: $length"

common_args=(
    --dataset_train ${traindata}
    --dataset_test ${testdata}
    --output_dir ${output_dir}
    --lr ${lr_rate}
    --num_train_epochs 20
    --batch_size 32
    --class_num 2
    --logging_steps 512
    --seed ${seed}
    --extra_eval dataset/m6a_data/miCLIP/${length}/unbalanced_test.fa
)

## RNAFM
echo "RNAFM"
python m6a_cls.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --use_kmer 1 \
    --pad_token_id 1 \
    "${common_args[@]}"

## RNAMSM
echo "RNAMSM"
python m6a_cls.py --method RNAMSM \
    --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
    --model_config 'model/configs/RNAMSM.json' \
    --use_kmer 1 \
    --pad_token_id 1 \
    "${common_args[@]}"

# RNAErnie
echo "RNAErnie"
python m6a_cls.py --method RNAErnie \
    --model_path 'model/pretrained/rnaernie' \
    --use_kmer 0 \
    "${common_args[@]}"

## RNABERT
echo "RNABERT"
python m6a_cls.py --method RNABERT --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' \
    --model_config 'model/RNABERT/RNABERT.json' \
    --use_kmer 1 \
    "${common_args[@]}"

## DNABERT
echo "DNABERT"
python m6a_cls.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --use_kmer 3 \
    "${common_args[@]}"

## DNABERT2
echo "DNABERT2"
python m6a_cls.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --use_kmer 0 \
    --pad_token_id 3 \
    "${common_args[@]}"

# SpliceBERT
echo "SpliceBERT"
python m6a_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --use_kmer 1 \
    "${common_args[@]}"

# NucleotideTransformer
python m6a_cls.py --method NucleotideTransformer \
    --model_path 'model/pretrained/NucleotideTransformer2' \
    --use_kmer 0 \
    "${common_args[@]}"

# GENA-LM
python m6a_cls.py --method GENA-LM-base \
    --model_path 'model/pretrained/GENA-LM/gena-lm-bert-base-t2t' \
    --use_kmer 0 \
    --pad_token_id 3 \
    "${common_args[@]}"

# GENA-LM
python m6a_cls.py --method GENA-LM-large \
    --model_path 'model/pretrained/GENA-LM/gena-lm-bert-large-t2t' \
    --use_kmer 0 \
    --pad_token_id 3 \
    "${common_args[@]}"

# UTR-LM
python m6a_cls.py --method UTRLM \
    --vocab_path 'model/vocabs/UTRLM.txt' \
    --model_path 'model/UTRLM/model.pt' \
    --use_kmer 1 \
    --pad_token_id 0 \
    "${common_args[@]}"


# DeepM6A
echo "DeepM6A"
python m6a_cls.py --method DeepM6A \
    --output_dir ${output_dir} \
    --dataset_train ${traindata} \
    --dataset_test ${testdata} \
    --batch_size 256 \
    --num_train_epochs 20 \
    --lr ${lr_rate} \
    --use_kmer 0

# bCNNMethylpred
echo "bCNNMethylpred"
python m6a_cls.py --method bCNNMethylpred \
    --output_dir ${output_dir} \
    --num_train_epochs 50 \
    --batch_size 32 \
    --dataset_train ${traindata} \
    --dataset_test ${testdata} \
    --lr ${lr_rate}