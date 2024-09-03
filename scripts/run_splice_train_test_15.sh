#!bash

# RNAFM
echo "RNAFM"
python splice_cls.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --num_train_epochs 5 \
    --class_num 18 \
    --batch_size 12 \
    --output_dir 'model/fine_tuned/Splicing/RNAFM_15class' \
    --dataset_train 'dataset/splice_data/gtex_500_15tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_15tis/dataset_test.h5' \
    --use_kmer 1

# SpliceBERT
echo "SpliceBERT"
python splice_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --num_train_epochs 5 \
    --class_num 18 \
    --batch_size 12 \
    --output_dir 'model/fine_tuned/Splicing/SpliceBERT_15class' \
    --dataset_train 'dataset/splice_data/gtex_500_15tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_15tis/dataset_test.h5' \
    --use_kmer 1


## RNAMSM
echo "RNAMSM"
python splice_cls.py --method RNAMSM \
    --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
    --model_config 'model/configs/RNAMSM.json' \
    --num_train_epochs 5 \
    --class_num 18 \
    --batch_size 12 \
    --output_dir 'model/fine_tuned/Splicing/RNAMSM_15class' \
    --dataset_train 'dataset/splice_data/gtex_500_15tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_15tis/dataset_test.h5' \
    --use_kmer 1
    
echo "RNAErnie"
python splice_cls.py --method RNAErnie \
    --model_path 'model/pretrained/RNAErnie' \
    --num_train_epochs 5 \
    --class_num 18 \
    --batch_size 12 \
    --output_dir 'model/fine_tuned/Splicing/RNAErnie_15class' \
    --dataset_train 'dataset/splice_data/gtex_500_15tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_15tis/dataset_test.h5' \
    --use_kmer 0 \
    --logging_steps 10