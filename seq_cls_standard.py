import argparse
from transformers import AutoTokenizer
from evaluator import seq_cls_evaluator
from dataset import NT_splice_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer
from seq_cls import str2list
import torch

MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 1024,
               "RNAFM": 1024,
               'DNABERT': 512,
               'DNABERT2': 512,
               "SpliceBERT": 512,
               "RNAErnie": 512,
               "GENA-LM-base": 600//5,
               "GENA-LM-large": 600//5,
               "UTRLM": 1024,
               'NucleotideTransformer': 102,  # 6-mers,
               }
available_methods = MAX_SEQ_LEN.keys()

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
    parser.add_argument("--class_num", default=2, type=int)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--output_dir", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=1, type=int)
    parser.add_argument("--pad_token_id", default=0, type=int)
    parser.add_argument("--dataset_train", type=str)
    parser.add_argument("--dataset_test", type=str)
    parser.add_argument("--data_group", type=str, default='index')
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc,classwise_acc,classwise_prauc",)  # optional: Emb
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--extract_emb", default=False)

    args = parser.parse_args()

    ###
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ###

    if 'NT_splice_sites' in args.data_group:
        args.labelset = 'index'
        if args.data_group == 'NT_splice_sites_all':
            MAX_SEQ_LEN[args.method] = min(MAX_SEQ_LEN[args.method], 402) # 400 + 2tokens
        elif args.data_group == 'NT_splice_sites_acceptors' or args.data_group == 'NT_splice_sites_donors':
            MAX_SEQ_LEN[args.method] = min(MAX_SEQ_LEN[args.method], 602) # 600 + 2tokens
    else:
        raise NotImplementedError(
            f"data_group {args.data_group} not implemented yet.")

    args.max_seq_len = MAX_SEQ_LEN[args.method]

    dataset_train = NT_splice_dataset.SeqClsDataset(
        file_path=args.dataset_train)
    dataset_test = NT_splice_dataset.SeqClsDataset(
        file_path=args.dataset_test, train=False)

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