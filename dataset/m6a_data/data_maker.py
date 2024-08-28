# A data maker for m6a data.

# 原始数据包含来自miCLIP-Seq和来自m6A-Seq数据的两部分
# https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2516-4#Tab1

# 原始数据中，每条序列的长度为101bp，可能不适合需要长序列输入的模型。因此，使用data_maker创建101bp长度和509bp长度的序列数据，分别用作测试

import os
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyfaidx import Fasta


def parse_file(finput, positive_label):
    with open(finput) as f:
        pos_seqs = list(SeqIO.parse(f, 'fasta'))
        res = []
        for seq in pos_seqs:
            label = seq.id
            label = label.split('::')[-1]
            # the label is formated like '1:169708551-169708652(-)'
            # we need to split into chrom pos strand
            parts = label.replace('(', ':').split(':')
            # chromosome
            chrom = parts[0]
            # position
            start, end = parts[1].split('-')
            start = int(start) + 1
            end = int(end)
            #
            strand = parts[2][0]
            seq = str(seq.seq)
            ref_seq = str(fasta.get_seq('chr'+chrom, start=start, end=end,
                                        rc=(strand == '-'))).upper()
            assert ref_seq == seq

            pos = (start + end) / 2
            assert int(pos) == pos

            # extract new sequence
            L = int(pos - args.length)
            R = int(pos + args.length)
            new_label = f'{chrom}:{L}-{R}:{strand}:{positive_label}'
            new_seq = str(fasta.get_seq('chr'+chrom, start=L, end=R,
                                        rc=(strand == '-'))).upper()
            res.append(SeqRecord(new_seq, id=new_label, description=""))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("--input", default=6, type=str)
    parser.add_argument("--output_folder", default=2, type=str)
    parser.add_argument("--length", default=101, type=int)
    parser.add_argument(
        "--genome", default='dataset/splice_data/required_files/hg38.fa', type=str)

    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fasta = Fasta(args.genome)

    # parse miCLIP-seq data
    folder = os.path.join(output_folder, 'miCLIP')
    os.makedirs(folder, exist_ok=True)
    seqs = parse_file(os.path.join(input_folder, 'train_pos.fa'), 1) + \
        parse_file(os.path.join(input_folder, 'train_neg.fa'), 0)
    SeqIO.write(seqs, os.path.join(folder, 'train.fa'), "fasta-2line")
    seqs = parse_file(os.path.join(input_folder, 'test_pos.fa'), 1) + \
        parse_file(os.path.join(input_folder, 'test_neg.fa'), 0)
    SeqIO.write(seqs, os.path.join(folder, 'test.fa'), "fasta-2line")

    # parse m6A-seq data

    folder = os.path.join(output_folder, 'm6A')
    os.makedirs(folder, exist_ok=True)
    seqs = parse_file(os.path.join(input_folder, 'hepg2_brain', 'test_pos.fa'), 1) + \
        parse_file(os.path.join(input_folder, 'hepg2_brain', 'test_neg.fa'), 0)
    SeqIO.write(seqs, os.path.join(folder, 'test.fa'), "fasta-2line")
    pass
