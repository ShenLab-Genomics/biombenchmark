import argparse
from transformers import AutoTokenizer
from evaluator import m6a_evaluator
from dataset import m6a_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer

MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 512,
               "RNAFM": 512,
               'DNABERT': 512,
               "SpliceBERT": 512,
               "RNAErnie": 512,
               "DeepM6A": 101
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
        "--method", choices=['RNAErnie', 'RNAFM', 'RNAMSM', 'DNABERT', 'SpliceBERT', 'SpliceAI', 'SpTransformer', 'DeepM6A', 'RNABERT'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--class_num", default=3, type=int)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=0, type=int)
    parser.add_argument("--dataset_train", type=str)
    parser.add_argument("--dataset_test", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc,pr_auc",)

    args = parser.parse_args()

    dataset_train = m6a_dataset.M6ADataset(
        fasta_dir=args.dataset_train)
    dataset_test = m6a_dataset.M6ADataset(
        fasta_dir=args.dataset_test)

    if (args.method == 'SpliceBERT') or (args.method == 'DNABERT'):
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        ev = m6a_evaluator.DNABERTEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAFM':
        args.replace_T = True
        args.replace_U = False
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        tokenizer = RNATokenizer(args.vocab_path)

        ev = m6a_evaluator.RNAFMEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'DeepM6A':
        args.replace_T = False
        args.replace_U = True
        args.max_seq_len = MAX_SEQ_LEN[args.method]

        ev = m6a_evaluator.DeepM6ASeqEvaluator(args, tokenizer=None)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNABERT':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        ev = m6a_evaluator.RNABertEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)
