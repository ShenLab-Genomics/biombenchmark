import argparse
from transformers import AutoTokenizer
from evaluator import mrl_evaluator
from dataset import mrl_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer

MAX_SEQ_LEN = {"RNABERT": 100,
               "RNAMSM": 100,
               "RNAFM": 100,
               'DNABERT': 100,
               'DNABERT2': 100,
               "SpliceBERT": 100,
               "RNAErnie": 100,
               "DeepM6A": 100,
               "UTRLM": 100,
               "PureResNet": 100
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
        "--method", choices=['RNAErnie', 'RNAFM', 'RNAMSM', 'DNABERT', 'DNABERT2', 'SpliceBERT', 'RNABERT', 'UTRLM', 'PureResNet'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=0, type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="MAE,MSE,R2,pearson,spearman")

    args = parser.parse_args()

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

    if args.method == 'PureResNet':
        args.replace_T = True
        args.replace_U = False
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        tokenizer = RNATokenizer(args.vocab_path)

        ev = mrl_evaluator.ResNetEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNABERT':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = mrl_evaluator.RNABertEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAMSM':
        args.max_seq_len = MAX_SEQ_LEN["RNAMSM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = mrl_evaluator.RNAMsmEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'UTRLM':
        args.max_seq_len = MAX_SEQ_LEN["UTRLM"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = RNATokenizer(args.vocab_path)

        # ev = mrl_evaluator.UTRLMEvaluator(args, tokenizer=None)
        # ev.run(args, dataset_train, dataset_test)
        ev = mrl_evaluator.UTRLMoriginalEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)
