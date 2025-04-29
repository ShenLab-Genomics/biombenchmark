#!bash

lr_rate=1e-4
freeze=0

dataset=dataset/mrl_data/mpra_data_varlen.csv
output_dir=logs/mrl_${lr_rate}_freeze${freeze}

echo "Learning rate set to: $lr_rate"
echo "Freeze base: $freeze"

common_args=(
    --dataset ${dataset}
    --output_dir ${output_dir}
    --lr ${lr_rate}
    --num_train_epochs 30
    --batch_size 64
    --logging_steps 1024
    --freeze_base ${freeze}
)

python mrl_pred.py --method DNABERT \
    --model_path 'model/pretrained/DNABERT/DNABERT1' \
    --use_kmer 3 \
    "${common_args[@]}"

# python mrl_pred.py --method DNABERT2 \
#     --model_path 'model/pretrained/DNABERT/DNABERT-2-117M' \
#     --pad_token_id 3 \
#     --use_kmer 0 \
#     "${common_args[@]}"