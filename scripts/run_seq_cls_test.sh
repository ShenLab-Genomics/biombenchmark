#!bash

## RNABERT
python seq_cls.py --method RNABERT --vocab_path 'model/RNABERT/vocab.txt' \
    --model_path 'model/pretrained/RNABERT/RNABERT.pth' --model_config 'model/RNABERT/RNABERT.json' \
    --dataset dataset/seq_cls_data --num_train_epochs 2

# ## RNAMSM
# python seq_cls.py --method RNAMSM --vocab_path 'model/vocabs/RNAMSM.txt' \
#     --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' --model_config 'model/configs/RNAMSM.json' \
#     --dataset dataset/seq_cls_data --num_train_epochs 2

# ## RNAFM
# python seq_cls.py --method RNAFM --vocab_path 'model/vocabs/RNAFM.txt' \
#     --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
#     --dataset dataset/seq_cls_data --num_train_epochs 2

# ## DNABERT
# python seq_cls.py --method DNABERT \
#     --model_path 'model/pretrained/DNABERT/DNABERT1' \
#     --dataset dataset/seq_cls_data --num_train_epochs 2

# ## SpliceBERT
# python seq_cls.py --method SpliceBERT \
#     --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
#     --dataset dataset/seq_cls_data --num_train_epochs 2

