#!bash

# RNAFM
echo "RNAFM"
python splice_cls.py --method RNAFM \
    --vocab_path 'model/vocabs/RNAFM.txt' \
    --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
    --model_config 'model/configs/RNAFM.json' \
    --num_train_epochs 10 \
    --class_num 56 \
    --output_dir 'model/fine_tuned/Splicing/RNAFM' \
    --dataset_train 'dataset/splice_data/gtex_500_53tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_53tis/dataset_test.h5' \
    --use_kmer 1 \
    --logging_steps 500

# SpliceBERT
echo "SpliceBERT"
python splice_cls.py --method SpliceBERT \
    --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' \
    --num_train_epochs 10 \
    --class_num 56 \
    --output_dir 'model/fine_tuned/Splicing/SpliceBERT' \
    --dataset_train 'dataset/splice_data/gtex_500_53tis/dataset_train_debug.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_53tis/dataset_test_debug.h5' \
    --use_kmer 1 \
    --logging_steps 500
    
## DNABERT
# echo "DNABERT"
# python splice_cls.py --method DNABERT \
#     --model_path 'model/pretrained/DNABERT/DNABERT1' \
#     --num_train_epochs 10 \
#     --class_num 18 \
#     --output_dir 'model/fine_tuned/Splicing/DNABERT1' \
#     --dataset_train 'dataset/splice_data/gtex_500_15tis/dataset_train_debug.h5' \
#     --dataset_test 'dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5' \
#     --use_kmer 1


# echo "RNAErnie"
# python splice_cls.py --method RNAErnie \
#     --model_path 'model/pretrained/RNAErnie' \
#     --num_train_epochs 10 \
#     --class_num 18 \
#     --output_dir 'model/fine_tuned/Splicing/RNAErnie' \
#     --dataset_train 'dataset/splice_data/gtex_500_15tis/dataset_train_debug.h5' \
#     --dataset_test 'dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5' \
#     --use_kmer 0 \
#     --logging_steps 10