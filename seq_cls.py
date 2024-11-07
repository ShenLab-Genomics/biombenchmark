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
    parser.add_argument("--pad_token_id", default=0, type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc",)  # optional: Emb
    parser.add_argument("--extract_emb", default=False)

    args = parser.parse_args()

    ###
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ###

    dataset_train = seq_cls_dataset.SeqClsDataset(
        fasta_dir=args.dataset, prefix='nRC')
    dataset_test = seq_cls_dataset.SeqClsDataset(
        fasta_dir=args.dataset, prefix='nRC', train=False)

    if args.method == 'RNABERT':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = seq_cls_evaluator.RNABertEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAFM':
        args.max_seq_len = MAX_SEQ_LEN["RNAFM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = seq_cls_evaluator.RNAFMEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAMSM':
        args.max_seq_len = MAX_SEQ_LEN["RNAMSM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = seq_cls_evaluator.RNAMsmEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'DNABERT':
        args.max_seq_len = MAX_SEQ_LEN["DNABERT"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = seq_cls_evaluator.DNABERTEvaluatorSeqCls(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'SpliceBERT':
        args.max_seq_len = MAX_SEQ_LEN["SpliceBERT"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = seq_cls_evaluator.SpliceBERTEvaluatorSeqCls(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAErnie':
        args.max_seq_len = MAX_SEQ_LEN["RNAErnie"]
        args.replace_T = True
        args.replace_U = False

        tokenizer = RNAErnieTokenizer.from_pretrained(
            args.model_path,
        )
        ev = seq_cls_evaluator.RNAErnieEvaluator(
            args, tokenizer=tokenizer)  # load tokenizer from model
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'DNABERT2':
        args.max_seq_len = MAX_SEQ_LEN["DNABERT2"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = seq_cls_evaluator.DNABERT2Evaluator(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)
