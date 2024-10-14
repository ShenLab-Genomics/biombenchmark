#!bash

echo "DNABERT2 - 509bp - miCLIP"
python m6a_cls.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/miCLIP/test.fa' \
    --num_train_epochs 10 \
    --batch_size 32 \
    --class_num 2 \
    --logging_steps 50 \
    --use_kmer 0

echo "DNABERT2 - 509bp - m6A"
python m6a_cls.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --dataset_train 'dataset/m6a_data/509bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/509bp/m6A/test.fa' \
    --num_train_epochs 10 \
    --batch_size 32 \
    --class_num 2 \
    --logging_steps 50 \
    --use_kmer 0

echo "DNABERT2 - 101bp - m6A"
python m6a_cls.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --dataset_train 'dataset/m6a_data/101bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/101bp/m6A/test.fa' \
    --num_train_epochs 10 \
    --batch_size 32 \
    --class_num 2 \
    --logging_steps 50 \
    --use_kmer 0


echo "DNABERT2 - 101bp - miCLIP"
python m6a_cls.py --method DNABERT2 \
    --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
    --dataset_train 'dataset/m6a_data/101bp/miCLIP/train.fa' \
    --dataset_test 'dataset/m6a_data/101bp/miCLIP/test.fa' \
    --num_train_epochs 10 \
    --batch_size 32 \
    --class_num 2 \
    --logging_steps 50 \
    --use_kmer 0