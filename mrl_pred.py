import argparse
from transformers import AutoTokenizer
from evaluator import mrl_evaluator
from dataset import mrl_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer
import torch

MAX_SEQ_LEN = {"RNABERT": 102,
               "RNAMSM": 102,
               "RNAFM": 102,
               'DNABERT': 102,
               'DNABERT2': 102,
               "SpliceBERT": 102,
               "RNAErnie": 102,
               "DeepM6A": 102,
               "ResNet": 102,
               'Optimus': 101,
               'NucleotideTransformer': 25,
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
        "--method", choices=['RNAErnie', 'RNAFM', 'RNAMSM', 'DNABERT', 'DNABERT2', 'SpliceBERT', 'RNABERT', 'ResNet', 'Optimus','NucleotideTransformer'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=0, type=int)
    parser.add_argument("--pad_token_id", default=0, type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="MAE,MSE,R2,pearson,spearman")

    args = parser.parse_args()

    ###
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ###

    dataset_train = mrl_dataset.MRLDataset(
        args.dataset, split_name="train")
    dataset_test = mrl_dataset.MRLDataset(
        args.dataset, split_name="test")

    if args.method == 'RNAFM':
        args.replace_T = True
        args.replace_U = False
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        tokenizer = RNATokenizer(args.vocab_path)

        ev = mrl_evaluator.RNAFMEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAErnie':
        args.max_seq_len = MAX_SEQ_LEN["RNAErnie"]
        args.replace_T = True
        args.replace_U = False

        tokenizer = RNAErnieTokenizer.from_pretrained(
            args.model_path,
        )
        ev = mrl_evaluator.RNAErnieEvaluator(
            args, tokenizer=tokenizer)  # load tokenizer from model
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'ResNet':
        args.replace_T = True
        args.replace_U = False
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        tokenizer = RNATokenizer(args.vocab_path)

        ev = mrl_evaluator.ResNetEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAMSM':
        args.max_seq_len = MAX_SEQ_LEN["RNAMSM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = mrl_evaluator.RNAMsmEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNABERT':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = mrl_evaluator.RNABertEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if (args.method == 'SpliceBERT') or (args.method == 'DNABERT'):
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = mrl_evaluator.DNABERTEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'DNABERT2':
        args.max_seq_len = MAX_SEQ_LEN["DNABERT2"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = mrl_evaluator.DNABERT2Evaluator(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'Optimus':
        args.max_seq_len = MAX_SEQ_LEN["Optimus"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = mrl_evaluator.OptimusEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'NucleotideTransformer':
        args.max_seq_len = MAX_SEQ_LEN["NucleotideTransformer"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = mrl_evaluator.NTEvaluator(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)
