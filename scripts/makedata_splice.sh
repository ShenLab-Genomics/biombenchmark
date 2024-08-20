#!bash


## 制作数据集
# python dataset/splice_data/data_maker.py --mode test_debug -c dataset/splice_data/configs_15tissue.yaml

# python dataset/splice_data/data_maker.py --mode train_debug -c dataset/splice_data/configs_15tissue.yaml

# python dataset/splice_data/data_maker.py --mode test -c dataset/splice_data/configs_15tissue.yaml

# python dataset/splice_data/data_maker.py --mode train -c dataset/splice_data/configs_15tissue.yaml

python dataset/splice_data/data_maker.py --mode test -c dataset/splice_data/configs_all_tissue.yaml

python dataset/splice_data/data_maker.py --mode train -c dataset/splice_data/configs_all_tissue.yaml