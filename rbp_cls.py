import argparse
from transformers import AutoTokenizer
from evaluator import rbp_evaluator
from transformers import AutoTokenizer
from model.BERT_like import RNATokenizer
from dataset import RBP_dataset

MAX_SEQ_LEN = {"RNAMSM": 512,
               "RNAFM": 512,
               'DNABERT': 512,
               "SpliceBERT": 512,
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
        "--method", choices=['RNAFM', 'RNAMSM', 'DNABERT', 'SpliceBERT', 'SpliceAI', 'SpTransformer'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--class_num", default=3)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--output_dir", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=1)
    parser.add_argument("--dataset_train", type=str)
    parser.add_argument("--dataset_test", type=str)
    parser.add_argument("--dataset", type=str, default='RBP')
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc",)

    args = parser.parse_args()
    if args.method == 'RNAFM':
        args.max_seq_len = MAX_SEQ_LEN["RNAFM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
        dataset_train = RBP_dataset.RBPDataset(
            fasta_path=args.dataset_train)
        dataset_eval = RBP_dataset.RBPDataset(
            fasta_path=args.dataset_test)

        ev = rbp_evaluator.RNAFMEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_eval)

    if args.method == 'RNAMSM':
        args.max_seq_len = MAX_SEQ_LEN["RNAMSM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
        dataset_train = RBP_dataset.RBPDataset(
            fasta_path=args.dataset_train)
        dataset_eval = RBP_dataset.RBPDataset(
            fasta_path=args.dataset_test)
        ev = rbp_evaluator.RNAMsmEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_eval)
