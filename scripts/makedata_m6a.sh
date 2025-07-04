# python dataset/m6a_data/data_maker.py --input dataset/m6a_data/source_data --output_folder dataset/m6a_data/509bp --length 254

# python dataset/m6a_data/data_maker.py --input dataset/m6a_data/source_data --output_folder dataset/m6a_data/101bp --length 50

python dataset/m6a_data/data_maker.py --input dataset/m6a_data/source_data --output_folder dataset/m6a_data/509bp/miCLIP --length 254 --unbalanced_test

python dataset/m6a_data/data_maker.py --input dataset/m6a_data/source_data --output_folder dataset/m6a_data/101bp/miCLIP --length 50 --unbalanced_test