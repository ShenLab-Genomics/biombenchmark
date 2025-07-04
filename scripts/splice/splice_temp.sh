#!bash

lr_rate=1e-4
class_num=3
echo "Learning rate set to: $lr_rate"

trainset='dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5'
testset='dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5'

common_args=(
    --output_dir model/fine_tuned/Splicing/${class_num}class_${lr_rate}_small
    --num_train_epochs 10
    --batch_size 12
    --logging_steps 512
    --class_num ${class_num}
    --lr ${lr_rate}
    --dataset_train ${trainset} \
    --dataset_test ${testset} \
)

# echo "UTRLM"
# python splice_cls.py --method UTRLM \
#     --vocab_path 'model/vocabs/UTRLM.txt' \
#     --model_path 'model/UTRLM/model.pt' \
#     --use_kmer 1 \
#     --pad_token_id 0 \
#     "${common_args[@]}"

# echo "GENA-LM-base"
# python splice_cls.py --method GENA-LM-base \
#     --model_path 'model/pretrained/GENA-LM/gena-lm-bert-base-t2t' \
#     --use_kmer 0 \
#     --pad_token_id 3 \
#     "${common_args[@]}"

echo "HyenaDNA_short"
python splice_cls.py --method HyenaDNA_short \
    --model_path 'model/pretrained/hyenadna-small-32k-seqlen' \
    --use_kmer 0 \
    --pad_token_id 4 \
    "${common_args[@]}"

# echo "HyenaDNA"
# python splice_cls.py --method HyenaDNA \
#     --model_path 'model/pretrained/hyenadna-small-32k-seqlen' \
#     --use_kmer 0 \
#     --pad_token_id 4 \
#     "${common_args[@]}"