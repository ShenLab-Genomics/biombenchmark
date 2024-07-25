import os
import sys
import pandas as pd
import numpy as np
import yaml

from tqdm import tqdm

chr_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6',
            'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
            'chr13', 'chr14', 'chr15', 'chr16', 'chr16', 'chr17', 'chr18',
            'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']


def build_sample_list():
    sample_df = pd.read_csv(sample_list, delimiter='\t')
    sample_df = sample_df[['SAMPID', 'SMTS', 'SMTSD', 'SMMPPD']]
    sample_df = sample_df[~sample_df['SAMPID'].str.contains('K-562')]
    sample_df = sample_df[sample_df['SMTSD'] !=
                          'Cells - EBV-transformed lymphocytes']
    target_file = os.path.join(data_path, 'sample.csv')
    sample_df.to_csv(target_file, index=None)
    pass


def get_paralog_list():
    para_df = pd.read_csv(para_db, header=0, sep='\t')
    para_df['paralogs'] = para_df['Human paralogue gene stable ID'].notnull()
    para_df['paralogs'] = para_df['paralogs'].map({True: 1, False: 0})
    para_df = para_df[['Gene stable ID', 'paralogs']]
    para_df.columns = ['Description', 'paralog']
    para_df = para_df.drop_duplicates('Description', keep='first')
    return para_df


def filt_tissue_data_RPKM(df: pd.DataFrame, tissue_type_list):
    sample_file = os.path.join(data_path, 'sample.csv')
    sample_list = pd.read_csv(sample_file)
    sample_list = sample_list[~sample_list['SMMPPD'].isna()]
    all_samples = sample_list['SAMPID'].values
    columns = list(df.columns)
    all_samples = list(set(all_samples).intersection(set(columns)))
    tag_list = []
    #
    print('in')
    pdd_dict = {}
    for idx, row in sample_list.iterrows():
        pdd_dict[row['SAMPID']] = row['SMMPPD']
    #
    tqdm.pandas(desc="get feature in input_data")
    df[all_samples] = df[all_samples] * 1e9
    df[all_samples] = df.progress_apply(
        lambda row: row[all_samples] / (row['right']-row['left']+1), axis=1)
    for sample in tqdm(list(all_samples)):
        df[sample] = df[sample] / pdd_dict[sample]

    for tissue in tissue_type_list:
        samples = sample_list[sample_list['SMTSD']
                              == tissue]['SAMPID'].values
        samples = list(set(samples).intersection(all_samples))
        tag = 'uL_'+str(tissue)
        tag_list.append(tag)
        df[tag] = df.groupby('left')[samples].transform(
            'sum').sum(axis=1)
        # df[tag] = df[tag] / (df[tag].max())
        tag = 'uR_'+str(tissue)
        tag_list.append(tag)
        df[tag] = df.groupby('right')[samples].transform(
            'sum').sum(axis=1)
        # df[tag] = df[tag] / (df[tag].max())
    return df[tag_list], tag_list


def filt_tissue_data(df: pd.DataFrame, tissue_type_list, class_type='detail'):
    '''
    class_type = 'detail' | 'group'
        'detail' means using detailed tissue name
        'group' using rough tissue type
    '''
    sample_file = os.path.join(data_path, 'sample.csv')
    sample_list = pd.read_csv(sample_file)
    all_samples = sample_list['SAMPID'].values
    columns = list(df.columns)
    all_samples = set(all_samples).intersection(set(columns))
    tag_list = []
    for tissue in tissue_type_list:
        if class_type == 'detail':
            samples = sample_list[sample_list['SMTSD']
                                  == tissue]['SAMPID'].values
        elif class_type == 'group':
            samples = sample_list[sample_list['SMTS']
                                  == tissue]['SAMPID'].values
        samples = list(set(samples).intersection(all_samples))
        tag = 'uL_'+str(tissue)
        tag_list.append(tag)
        df[tag] = (df.groupby('left')[samples].transform(
            'sum') > 0).sum(axis=1) / len(samples)
        tag = 'uR_'+str(tissue)
        tag_list.append(tag)
        df[tag] = (df.groupby('right')[samples].transform(
            'sum') > 0).sum(axis=1) / len(samples)
    return df[tag_list], tag_list


def cut_description(df):
    s = str(df['Description'])
    df['Description'] = s[0:s.find('.')]
    return df


def get_merged_infomation(data_folder, output_folder, tissue_list, class_type):
    gene_df = pd.read_csv(gene_db, header=0, sep='\t')
    gene_df = gene_df[['Gene stable ID', 'Gene name',
                       'Strand']].drop_duplicates(keep='first')
    gene_df['Description'] = gene_df['Gene stable ID']
    for chrom in chr_list:
        fpath = os.path.join(data_folder, chrom+'.csv')
        df = pd.read_csv(fpath, header=0)
        df_data, tag_list = filt_tissue_data(
            df, tissue_list, class_type=class_type)
        df = df[['chr', 'left', 'right', 'Description']]
        df = pd.concat([df, df_data], axis=1)
        df = df.apply(cut_description, axis=1)
        para_df = get_paralog_list()
        df = df.merge(para_df, on='Description', how='inner')
        df = df.merge(gene_df, on='Description', how='inner')
        df = df.drop_duplicates(['left', 'right'], keep='first')
        foutpath = os.path.join(output_folder, chrom+'_after.csv')
        df = df[['left', 'right', 'Description', 'Gene name',
                 'Strand', 'paralog']+tag_list]
        df.columns = ['left', 'right', 'description',
                      'gene', 'strand', 'paralog']+tag_list
        df.to_csv(foutpath, index=True)


if __name__ == '__main__':
    CONFIG_PATH = sys.argv[1]
    with open(CONFIG_PATH, encoding='utf-8') as config:
        configs = yaml.load(config, Loader=yaml.FullLoader)
    print(configs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_list = configs['source_data']['sample_list']
    gene_db = configs['source_data']['gene_strand_data']
    para_db = configs['source_data']['paralog_data']
    data_path = configs['source_data']['data_path']
    sample_list = os.path.join(script_dir, sample_list)
    gene_db = os.path.join(script_dir, gene_db)
    para_db = os.path.join(script_dir, para_db)
    data_path = os.path.join(script_dir, data_path)
    output_folder = os.path.join(script_dir, configs['tissue_data_folder'])
    class_type = configs['class_type']

    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    build_sample_list()
    tissue_list = list(configs['tissue_dict_rev'].keys())
    get_merged_infomation(data_path, output_folder, tissue_list, class_type)
