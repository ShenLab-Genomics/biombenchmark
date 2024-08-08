import argparse
from transformers import AutoTokenizer
from evaluator import seq_cls_evaluator
from dataset import seq_cls_dataset
from model.BERT_like import RNATokenizer

MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 512,
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
        "--method", choices=['RNABERT', 'RNAFM', 'RNAMSM', 'DNABERT', 'SpliceBERT'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc",)

    args = parser.parse_args()

    if args.method == 'RNABERT':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
        dataset_train = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer)
        dataset_eval = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer, train=False)

        ev = seq_cls_evaluator.RNABertEvaluator(args)
        ev.run(args, dataset_train, dataset_eval)

    if args.method == 'RNAFM':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
        dataset_train = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer)
        dataset_eval = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer, train=False)

        ev = seq_cls_evaluator.RNAFMEvaluator(args)
        ev.run(args, dataset_train, dataset_eval)

    if args.method == 'RNAMSM':
        args.max_seq_len = MAX_SEQ_LEN["RNAMSM"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = RNATokenizer(args.vocab_path)
        dataset_train = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer)
        dataset_eval = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer, train=False)

        ev = seq_cls_evaluator.RNAMsmEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_eval)

    if args.method == 'DNABERT':
        args.max_seq_len = MAX_SEQ_LEN["DNABERT"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)
        dataset_train = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer)
        dataset_eval = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer, train=False)

        ev = seq_cls_evaluator.DNABERTEvaluatorSeqCls(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_eval)

    if args.method == 'SpliceBERT':
        args.max_seq_len = MAX_SEQ_LEN["SpliceBERT"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)
        dataset_train = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer)
        dataset_eval = seq_cls_dataset.SeqClsDataset(
            fasta_dir=args.dataset, prefix='nRC', tokenizer=tokenizer, train=False)

        ev = seq_cls_evaluator.SpliceBERTEvaluatorSeqCls(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_eval)
