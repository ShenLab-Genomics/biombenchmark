import argparse
from transformers import AutoTokenizer
from evaluator import seq_cls_evaluator
from dataset import seq_cls_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer
import torch

MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 512,
               "RNAFM": 512,
               'DNABERT': 512,
               'DNABERT2': 512,
               "SpliceBERT": 512,
               "RNAErnie": 512,
               "GENA-LM-base": 512//5,
               "GENA-LM-large": 512//5,
               "UTRLM": 512,
               'NucleotideTransformer': 100,  # 6-mers，
               "ncRDense": 750
               }
available_methods = MAX_SEQ_LEN.keys()


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
        "--method", choices=available_methods, default='RNABERT')
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
    parser.add_argument("--pad_token_id", default=0, type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_group", type=str, default='nRC')
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc,classwise_acc,classwise_prauc",)  # optional: Emb
    parser.add_argument("--extract_emb", default=False)
    parser.add_argument("--seed", default=2024, type=int)

    args = parser.parse_args()

    assert args.output_dir, "output_dir is required."

    ###
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ###
    if 'nRC' in args.data_group:
        args.labelset = 'nRC'
    elif 'mix' in args.data_group:
        args.labelset = 'mix'
    else:
        raise NotImplementedError(
            f"data_group {args.data_group} not implemented yet.")

    if args.method != 'ncRDense':
        dataset_train = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix=args.data_group)
        dataset_test = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix=args.data_group, train=False)
    else:
        dataset_train = seq_cls_dataset.SeqClsDatasetOneHot(
            fasta_dir=args.dataset, prefix=args.data_group, rnafold=True)
        dataset_test = seq_cls_dataset.SeqClsDatasetOneHot(
            fasta_dir=args.dataset, prefix=args.data_group, train=False, rnafold=True)

    #

    args.max_seq_len = MAX_SEQ_LEN[args.method]
    if args.method == 'RNABERT':
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = seq_cls_evaluator.RNABertEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAFM':
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = seq_cls_evaluator.RNAFMEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAMSM':
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = seq_cls_evaluator.RNAMsmEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'DNABERT':
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = seq_cls_evaluator.DNABERTEvaluatorSeqCls(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'SpliceBERT':
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = seq_cls_evaluator.SpliceBERTEvaluatorSeqCls(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAErnie':
        args.replace_T = True
        args.replace_U = False

        tokenizer = RNAErnieTokenizer.from_pretrained(
            args.model_path,
        )
        ev = seq_cls_evaluator.RNAErnieEvaluator(
            args, tokenizer=tokenizer)  # load tokenizer from model
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'DNABERT2':
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = seq_cls_evaluator.DNABERT2Evaluator(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'NucleotideTransformer':
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = seq_cls_evaluator.NTEvaluator(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'GENA-LM-base' or args.method == 'GENA-LM-large':
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = seq_cls_evaluator.GENAEvaluator(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'UTRLM':
        args.max_seq_len = MAX_SEQ_LEN["UTRLM"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = RNATokenizer(args.vocab_path)

        ev = seq_cls_evaluator.UTRLMEvaluator(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'ncRDense':
        args.max_seq_len = MAX_SEQ_LEN["ncRDense"]
        args.replace_T = False
        args.replace_U = True

        ev = seq_cls_evaluator.ncRDenseEvaluator(
            args, tokenizer=None)
        ev.run(args, dataset_train, dataset_test)
