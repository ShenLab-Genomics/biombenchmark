#!bash

## RNAFM
echo "RNAFM"
python seq_cls.py --method RNAFM --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --dataset dataset/seq_cls_data --num_train_epochs 30 \
    --batch_size 40


# ## RNABERT
# echo "RNABERT"
# python seq_cls.py --method RNABERT --vocab_path 'model/RNABERT/vocab.txt' \
#     --model_path 'model/pretrained/RNABERT/RNABERT.pth' --model_config 'model/RNABERT/RNABERT.json' \
#     --dataset dataset/seq_cls_data --num_train_epochs 30 \
#     --batch_size 50

# ## RNAMSM
# echo "RNAMSM"
# python seq_cls.py --method RNAMSM --vocab_path 'model/vocabs/RNAMSM.txt' \
#     --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' --model_config 'model/configs/RNAMSM.json' \
#     --dataset dataset/seq_cls_data --num_train_epochs 30 \
#     --batch_size 50


# ## DNABERT
# echo "DNABERT"
# python seq_cls.py --method DNABERT \
#     --model_path 'model/pretrained/DNABERT/DNABERT1' \
#     --dataset dataset/seq_cls_data --num_train_epochs 30 \
#     --logging_steps 500 \
#     --use_kmer 3 \
#     --batch_size 50

# ## SpliceBERT
# echo "SpliceBERT"
# python seq_cls.py --method SpliceBERT \
#     --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
#     --dataset dataset/seq_cls_data --num_train_epochs 30 \
#     --logging_steps 500 \
#     --batch_size 50

# # RNAErnie
# echo "RNAErnie"
# python seq_cls.py --method RNAErnie \
#     --vocab_path 'model/pretrained/RNAErnie/vocab.txt' \
#     --model_path 'model/pretrained/RNAErnie' \
#     --dataset dataset/seq_cls_data --num_train_epochs 30 \
#     --logging_steps 50 \
#     --use_kmer 1 \
#     --batch_size 50

# ## DNABERT2
# echo "DNABERT2"
# python seq_cls.py --method DNABERT2 \
#     --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
#     --dataset dataset/seq_cls_data --num_train_epochs 30 \
#     --logging_steps 50 \
#     --use_kmer 0 \
#     --batch_size 50