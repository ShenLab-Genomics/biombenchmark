#!bash

## RNAFM
echo "RNAFM"
python task_mrl_pred.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --num_train_epochs 30 \
    --batch_size 64 \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --use_kmer 1 \
    --lr 1e-3

echo "PureResNet"
python task_mrl_pred.py --method PureResNet \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_config 'model/configs/PureResNet.json' \
    --num_train_epochs 50 \
    --batch_size 64 \
    --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
    --use_kmer 1 \
    --lr 1e-3

## RNABERT
echo "RNABERT"
python task_mrl_pred.py --method RNABERT --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' \
    --model_config 'model/RNABERT/RNABERT.json' \
    --dataset 'dataset/mrl_data/GSM4084997_varying_length_25to100.csv' \
    --batch_size 32 \
    --num_train_epochs 5 \
    --logging_steps 20 \
    --lr 1e-4 \
    --use_kmer 1


# ## RNAMSM
# echo "RNAMSM"
# python task_mrl_pred.py --method RNAMSM \
#     --vocab_path 'model/vocabs/RNAMSM.txt' \
#     --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
#     --model_config 'model/configs/RNAMSM.json' \
#     --num_train_epochs 5 \
#     --batch_size 32 \
#     --dataset 'dataset/mrl_data/GSM4084997_varying_length_25to100.csv' \
#     --use_kmer 1 \
#     --lr 1e-4 


# echo "UTRLM"
# python task_mrl_pred.py --method UTRLM \
#     --vocab_path 'model/vocabs/UTRLM.txt' \
#     --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
#     --num_train_epochs 5 \
#     --batch_size 64 \
#     --dataset 'dataset/mrl_data/mpra_data_varlen.csv' \
#     --use_kmer 1 \
#     --lr 1e-3