import argparse
from transformers import AutoTokenizer
from evaluator import splice_evaluator
from dataset import splice_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer

MAX_SEQ_LEN = {"RNAMSM": 512,
               "RNAFM": 510,
               'DNABERT': 512,
               "SpliceBERT": 510,
               "RNAErnie": 1000,
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
        "--method", choices=['RNAErnie', 'RNAFM', 'RNAMSM', 'DNABERT', 'SpliceBERT', 'SpliceAI', 'SpTransformer'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--class_num", default=3, type=int)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--output_dir", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=0)
    parser.add_argument("--dataset_train", type=str)
    parser.add_argument("--dataset_test", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="topk,prauc,rocauc",)

    args = parser.parse_args()

    assert args.output_dir, "output_dir is required."

    if (args.method == 'SpliceBERT') or (args.method == 'DNABERT'):
        args.max_seq_len = MAX_SEQ_LEN[args.method]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        dataset_train = splice_dataset.SpliceNormalDataset(
            h5_filename=args.dataset_train)
        dataset_test = splice_dataset.SpliceNormalDataset(
            h5_filename=args.dataset_test)

        if args.method == 'SpliceBERT':
            ev = splice_evaluator.SpliceBERTEvaluator(
                args, tokenizer=tokenizer)
        elif args.method == 'DNABERT':
            ev = splice_evaluator.DNABERTEvaluator(
                args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAFM':
        args.max_seq_len = MAX_SEQ_LEN["RNAFM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)

        dataset_train = splice_dataset.SpliceNormalDataset(
            h5_filename=args.dataset_train)
        dataset_test = splice_dataset.SpliceNormalDataset(
            h5_filename=args.dataset_test)

        ev = splice_evaluator.RNAFMEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)

    if args.method == 'RNAErnie':
        args.max_seq_len = MAX_SEQ_LEN["RNAErnie"]
        args.replace_T = True
        args.replace_U = False
        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.model_path)
        tokenizer = RNAErnieTokenizer.from_pretrained(
            args.model_path,
        )

        dataset_train = splice_dataset.SpliceNormalDataset(
            h5_filename=args.dataset_train)
        dataset_test = splice_dataset.SpliceNormalDataset(
            h5_filename=args.dataset_test)

        ev = splice_evaluator.RNAErnieEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)
