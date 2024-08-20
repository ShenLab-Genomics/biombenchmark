#!bash

## RNAFM
# echo "RNAFM"
# python rbp_cls.py --method RNAFM --vocab_path 'model/vocabs/RNAFM.txt' \
#     --model_path 'model/pretrained/RNAFM/RNA-FM_pretrained.pth' \
#     --num_train_epochs 50 \
#     --dataset_train dataset/rbp_data/RBPdata1201/01_HITSCLIP_AGO2Karginov2013a_hg19/train/1/sequence.fa \
#     --dataset_test dataset/rbp_data/RBPdata1201/01_HITSCLIP_AGO2Karginov2013a_hg19/test/1/sequence.fa \

# ## RNAMSM
echo "RNAMSM"
python rbp_cls.py --method RNAMSM --vocab_path 'model/vocabs/RNAMSM.txt' \
    --model_path 'model/pretrained/RNAMSM/RNAMSM.pth' --model_config 'model/configs/RNAMSM.json' \
    --num_train_epochs 50 \
    --dataset_train dataset/rbp_data/RBPdata1201/01_HITSCLIP_AGO2Karginov2013a_hg19/train/1/sequence.fa \
    --dataset_test dataset/rbp_data/RBPdata1201/01_HITSCLIP_AGO2Karginov2013a_hg19/test/1/sequence.fa \
