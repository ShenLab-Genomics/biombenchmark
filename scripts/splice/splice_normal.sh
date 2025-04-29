#!bash

lr_rate=$1
class_num=3
trainset='dataset/splice_data/gtex_500_15tis/dataset_train.h5'
testset='dataset/splice_data/gtex_500_15tis/dataset_test.h5'

echo "Learning rate set to: $lr_rate"

common_args=(
    --output_dir model/fine_tuned/Splicing/${class_num}class_${lr_rate}
    --num_train_epochs 10
    --batch_size 12
    --logging_steps 512
    --class_num ${class_num}
    --lr ${lr_rate}
    --dataset_train ${trainset} \
    --dataset_test ${testset} \
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

## SpliceBERT
echo "SpliceBERT"
python seq_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    "${common_args[@]}"


echo "SpTransformer"
python splice_cls.py --method SpTransformer \
    --model_path 'model/pretrained/SpTransformer/weight.ckpt' \
    --use_kmer 0 \
    "${common_args[@]}"

echo "SpTransformer"
python splice_cls.py --method SpTransformer_short \
    --model_path 'model/pretrained/SpTransformer/weight.ckpt' \
    --use_kmer 0 \
    "${common_args[@]}"

# using special batch size
echo "NucleotideTransformer"
python seq_cls.py --method NucleotideTransformer \
    --model_path 'model/pretrained/NucleotideTransformer2' \
    --output_dir model/fine_tuned/Splicing/${class_num}class_${lr_rate} \
    --dataset_train ${trainset} \
    --dataset_test ${testset} \
    --num_train_epochs 10 \
    --batch_size 4 \
    --class_num ${class_num} \
    --lr ${lr_rate} \
    --use_kmer -10 

# using special batch size
echo "NucleotideTransformer-short"
python splice_cls.py --method NT_Short \
    --model_path 'model/pretrained/NucleotideTransformer2' \
    --output_dir model/fine_tuned/Splicing/${class_num}class_${lr_rate} \
    --dataset_train ${trainset} \
    --dataset_test ${testset} \
    --num_train_epochs 10 \
    --batch_size 4 \
    --class_num ${class_num} \
    --lr ${lr_rate} \
    --use_kmer -10 

echo "SpliceAI"
python splice_cls.py --method SpliceAI \
    --use_kmer 0 \
    "${common_args[@]}"

python splice_cls.py --method SpliceAI_short \
    --use_kmer 0
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

