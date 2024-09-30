#!bash

## RNAMSM
echo "RNAMSM"
python splice_cls.py --method RNAMSM \
    --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' \
    --model_config 'model/configs/RNAMSM.json' \
    --num_train_epochs 10 \
    --class_num 56 \
    --batch_size 12 \
    --output_dir 'model/fine_tuned/Splicing/RNAMSM_53class' \
    --dataset_train 'dataset/splice_data/gtex_500_53tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_53tis/dataset_test.h5' \
    --use_kmer 1

# SpTransformer
echo "SpTransformer"
python splice_cls.py --method SpTransformer \
    --model_path 'model/pretrained/SpTransformer/weight.ckpt' \
    --num_train_epochs 10 \
    --class_num 56 \
    --batch_size 12 \
    --output_dir 'model/fine_tuned/Splicing/SpTransformer_53class' \
    --dataset_train 'dataset/splice_data/gtex_500_53tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_53tis/dataset_test.h5' \
    --use_kmer 0 \
    --logging_steps 500

# RNAErnie
echo "RNAErnie"
python splice_cls.py --method RNAErnie \
    --model_path 'model/pretrained/RNAErnie' \
    --num_train_epochs 10 \
    --class_num 56 \
    --batch_size 12 \
    --output_dir 'model/fine_tuned/Splicing/RNAErnie_53class' \
    --dataset_train 'dataset/splice_data/gtex_500_53tis/dataset_train.h5' \
    --dataset_test 'dataset/splice_data/gtex_500_53tis/dataset_test.h5' \
    --use_kmer 0 \
    --logging_steps 500