# make splice data
import numpy as np
import pandas as pd
import os
import sys
import time
import h5py
import yaml
# from utils import name_decorate, one_hot_encode
import tqdm
from pyfaidx import Fasta
import argparse
from dataset.splice_data import paser

start_time = time.time()

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
# One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
# 2 is for donor and -1 is for padding.


def one_hot_encode(X, use_map):
    return use_map[X.astype('int8')]


class Site:
    def __init__(self, pos, usageL, usageR, jn_left=True, use_tissue=True):
        self.pos = int(pos)
        self.is_left = jn_left
        self.usage = [usageL, usageR]
        self.use_tissue = use_tissue
        pass

    def __eq__(self, other):
        if self.pos == other.pos and self.is_left == other.is_left:
            return True
        return False

    def __hash__(self):
        return hash(str(self.pos) + str(self.is_left))


def getseq(fasta, chrom, start, end, center, context):
    pad_L = context
    pad_R = context + (center - (end-start+1) % center)
    seq = fasta.get_seq(chrom, start, end)
    seq = 'N' * pad_L + str(seq) + 'N' * pad_R
    return seq, (center - (end-start+1) % center)


def make_sequence(site_list, fasta, chrom, rev=False, center=5000, context=5000, tissue_cnt=53, tx_start=None, tx_end=None):
    # gather all splice site from site_list, make one-hot data
    # slide_step = center//2
    # start = site_list[0].pos - slide_step
    start = site_list[0].pos
    end = site_list[-1].pos
    # print(start, end, (end-start+1))
    start = tx_start
    end = tx_end
    # print(start, end, (end-start+1))
    seq, offset = getseq(fasta, chrom, tx_start, tx_end, center,
                         context)  # 获取ref genome sequence
    # print(start, end, (end-start+1), len(seq))
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    Y1 = np.zeros((end - start + 1 + offset, tissue_cnt))  # for acceptor
    V1 = np.zeros((end - start + 1 + offset))  # visit flag
    if rev == False:  # 正链
        X0 = np.asarray(list(map(int, list(seq))))  # data
        Y0 = np.zeros(end - start + 1 + offset)  # label
        for c in site_list:
            if c.pos < start or c.pos > end:
                continue
            donor = c.is_left ^ rev
            index = 0 if c.is_left else 1
            if donor:
                Y0[c.pos-start] = 2
                # Y2[c.pos-start, :] += c.usage[index]
            else:
                Y0[c.pos-start] = 1
                # Y1[c.pos-start, :] += c.usage[index]
            if c.use_tissue:
                Y1[c.pos-start, :] = c.usage[index]
                V1[c.pos-start] = 1
            elif V1[c.pos-start] == 0:  # for gencode data which have no tissue usage label
                Y1[c.pos-start, :] = - np.ones((tissue_cnt))

    else:
        X0 = (5 - np.asarray(list(map(int, list(seq[::-1]))))) % 5
        Y0 = np.zeros(end - start + 1 + offset)  # label
        for c in site_list:
            if c.pos < start or c.pos > end:
                continue
            donor = c.is_left ^ rev
            index = 0 if c.is_left else 1
            # index = 1 if donor else 0
            if donor:
                Y0[offset + end-c.pos] = 2
                # Y2[offset + end-c.pos, :] += c.usage[index]
            else:
                Y0[offset + end-c.pos] = 1
                # Y1[offset + end-c.pos, :] += c.usage[index]
            if c.use_tissue:
                Y1[offset + end-c.pos, :] = c.usage[index]
                V1[offset + end-c.pos] = 1
            elif V1[offset + end-c.pos] == 0:
                Y1[offset + end-c.pos, :] = - np.ones((tissue_cnt))
    # 此时，X0长度应为 context + center * N + context
    # 以center//2为步长来移动

    # print(X0.shape, Y0.shape, Y1.shape)
    num_points = (X0.shape[0]-2*context)//center
    Xd = np.zeros((num_points, center + 2*context))
    Yd = np.zeros((num_points, center))
    Yd1 = np.zeros((num_points, center, tissue_cnt))  # acceptor usage
    # print(num_points)
    for i in range(num_points):
        Xd[i] = X0[center * i: 2*context + center * (i + 1)]
        # print(slide_step * i,  2*context + slide_step * (i + 2))
        # Xd[i] = X0[slide_step * i: 2*context + slide_step * (i + 2)]
    for i in range(num_points):
        Yd[i] = Y0[center * i: center * (i + 1)]
        # Yd[i] = Y0[slide_step * i: slide_step * (i + 2)]

        Yd1[i] = Y1[center * i: center * (i + 1)]
        # Yd1[i] = Y1[slide_step * i: slide_step * (i + 2)]

    # Xd, Yd = one_hot_encode(Xd, [Yd])
    Yd = one_hot_encode(Yd, OUT_MAP)
    Yd = np.concatenate([Yd, Yd1], axis=2)
    # Xd = Xd.transpose((0, 2, 1))  # for pytorch
    Yd = Yd.transpose((0, 2, 1))

    # discard sequences with no exon region
    not_empty = np.sum(Yd[:, 1:3, :], axis=(1, 2)) > 0

    Xd = Xd[not_empty]
    Yd = Yd[not_empty]
    return Xd, Yd


def tensor2numpy(input_tensor):
    # device cuda Tensor to host numpy
    return input_tensor.cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process some chromosomes based on the given mode.')

    # Adding the first argument for the operation mode
    parser.add_argument('--mode', choices=['train', 'test', 'train_debug', 'test_debug', 'all'],
                        help='The mode of operation. Can be train, test, test_single or all.')
    parser.add_argument(
        '-c', '--config', default='dataset/splice_data/configs_15tissue.yaml',
        help='The config file containing path of required files')

    # # Adding the second argument for the flag
    # parser.add_argument('flag', choices=['0', '1', 'all'],
    #                     help='A flag to control certain behaviors. Can be 0, 1 or all.')

    args = parser.parse_args()
    flag = 1
    if args.mode == 'train':
        CHROM_GROUP = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                       'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                       'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']
    elif args.mode == 'test':
        CHROM_GROUP = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        flag = 0

    elif args.mode == 'train_debug':
        CHROM_GROUP = ['chr2']

    elif args.mode == 'test_debug':
        CHROM_GROUP = ['chr1']
        flag = 0

    else:  # args.mode == 'all'
        CHROM_GROUP = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9',
                       'chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                       'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                       'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']

    with open(args.config, encoding='utf-8') as config:
        configs = yaml.load(config, Loader=yaml.FullLoader)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_folder = os.path.join(script_dir, configs['tissue_data_folder'])
    gene_db = configs['source_data']['gene_strand_data']
    gene_db = os.path.join(script_dir, gene_db)
    fasta = configs['ref_genome']
    fasta = os.path.join(script_dir, fasta)

    # Parse source data if needed
    check_exist = True
    for chrom in CHROM_GROUP:
        if not os.path.exists(os.path.join(data_folder, chrom+'_after.csv')):
            check_exist = False
            break
    if not check_exist:
        # parse source data
        print('Pre-processing required')
        os.makedirs(data_folder, exist_ok=True)
        paser.build_sample_list()
        tissue_list = list(configs['tissue_dict_rev'].keys())
        class_type = configs['class_type']
        data_path = configs['source_data']['data_path']
        data_path = os.path.join(script_dir, data_path)
        paser.get_merged_infomation(
            data_path, data_folder, tissue_list, class_type)
    print('Pre-processing finished')
    ##

    # You can access the arguments like this:
    print(f"Mode: {args.mode}, Flag: {flag}, Config:{args.config}")
    print(f"Chromosomes selected: {CHROM_GROUP}")

    TISSUE_CNT = len(configs['tissue_dict'])
    print(TISSUE_CNT)

    # load transcript data
    tx_df = pd.read_csv(gene_db, header=0, sep='\t')
    tx_df = tx_df[['Gene stable ID',
                   'Transcript start (bp)', 'Transcript end (bp)']]
    tx_df.columns = ['description', 'start', 'end']
    tx_df = tx_df.groupby('description').agg({'start': 'min', 'end': 'max'})
    ##

    h5f2 = h5py.File(os.path.join(data_folder,
                                  'dataset' + '_' +
                                  args.mode + '.h5'), 'w')

    fasta = Fasta(fasta)
    batch_cnt = 0
    for chrom in CHROM_GROUP:
        print('process:', chrom)
        # process GTEx csv
        fpath = os.path.join(data_folder, chrom+'_after.csv')
        df = pd.read_csv(fpath, header=0)
        tag_listL = [tag for tag in df.columns if str(tag)[0:2] == 'uL']
        tag_listR = [tag for tag in df.columns if str(tag)[0:2] == 'uR']
        df_group = df.groupby('description')
        for gene_id, gene_group in tqdm.tqdm(df_group, total=len(df_group), mininterval=3):
            posi_site_list = []
            nega_site_list = []
            X_batch = []
            Y_batch = []
            cnt = 0
            cnt_genL = 0
            cnt_genR = 0
            tx_start = tx_df.loc[gene_id]['start']
            tx_end = tx_df.loc[gene_id]['end']
            for idx, row in gene_group.iterrows():
                if row['left'] == row['right']:
                    # print('Skipped:{}'.format(row['left']))
                    continue
                usage_label = np.sum([row[tag_listL], row[tag_listR]])
                paralog = row['paralog']
                if ((paralog == 1) and (sys.argv[2] == '0')):
                    break
                cnt += 1
                if row['strand'] == 1:  # positive 正链
                    posi_site_list.append(
                        Site(row['left']-1, row[tag_listL], row[tag_listR], jn_left=True))
                    posi_site_list.append(
                        Site(row['right']+1, row[tag_listL], row[tag_listR], jn_left=False))
                else:
                    nega_site_list.append(
                        Site(row['left']-1, row[tag_listL], row[tag_listR], jn_left=True))
                    nega_site_list.append(
                        Site(row['right']+1, row[tag_listL], row[tag_listR], jn_left=False))

            posi_site_list = list(set(posi_site_list))
            nega_site_list = list(set(nega_site_list))
            posi_site_list = sorted(
                posi_site_list, key=lambda x: x.pos, reverse=False)
            nega_site_list = sorted(
                nega_site_list, key=lambda x: x.pos, reverse=False)
            if len(posi_site_list) > 0:  # 处理正链，区间覆盖
                Xd, Yd = make_sequence(posi_site_list, fasta, chrom, rev=False,
                                       center=configs['center_len'], context=configs['context_len'], tissue_cnt=TISSUE_CNT,
                                       tx_start=tx_start, tx_end=tx_end)
                X_batch.extend(Xd)
                Y_batch.extend(Yd)

            if len(nega_site_list) > 0:  # 处理反链，区间覆盖
                Xd, Yd = make_sequence(nega_site_list, fasta, chrom, rev=True,
                                       center=configs['center_len'], context=configs['context_len'], tissue_cnt=TISSUE_CNT,
                                       tx_start=tx_start, tx_end=tx_end)
                X_batch.extend(Xd)
                Y_batch.extend(Yd)

            batch_cnt += 1

            h5f2.create_dataset('X' + str(batch_cnt), data=X_batch)
            h5f2.create_dataset('Y' + str(batch_cnt), data=Y_batch)

            if ('debug' in args.mode) and (batch_cnt > 200):
                break
        flag = 0
    h5f2.close()
    pass