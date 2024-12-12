import argparse
from transformers import AutoTokenizer
from evaluator import seq_cls_evaluator
from dataset import seq_cls_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer

MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 512,
               "RNAFM": 512,
               'DNABERT': 512,
               'DNABERT2': 512,
               "SpliceBERT": 512,
               "RNAErnie": 512
               }


def str2list(v):
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        vs = v.split(",")
        return [v.strip() for v in vs]
    else:
        raise argparse.ArgumentTypeError(
            "Str value seperated by ', ' expected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        "--method", choices=['RNAErnie', 'RNABERT', 'RNAFM', 'RNAMSM', 'DNABERT', 'SpliceBERT', 'DNABERT2'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--class_num", default=13, type=int)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--output_dir", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=1, type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc",)  # optional: Emb

    args = parser.parse_args()

    dataset_train = seq_cls_dataset.SeqClsDataset(
        fasta_dir=args.dataset, prefix='nRC')
    dataset_test = seq_cls_dataset.SeqClsDataset(
        fasta_dir=args.dataset, prefix='nRC', train=False)

    if args.method == 'RNABERT':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
