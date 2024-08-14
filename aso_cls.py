import argparse
import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer
from evaluator import aso_pred_evaluator
from dataset import aso_pred_dataset
from model.BERT_like import RNATokenizer

from seq_cls import MAX_SEQ_LEN, str2list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        "--method", choices=['RNABERT', 'RNAFM', 'RNAMSM', 'DNABERT', 'SpliceBERT'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=1)
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc",)

    args = parser.parse_args()

    if args.method == 'RNABERT':
        args.max_seq_len = MAX_SEQ_LEN["RNABERT"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
        dataset = aso_pred_dataset.ASOTokenDataset(
            fpath=args.dataset)
        rand_indices = torch.randperm(len(dataset)).tolist()

        dataset_train = Subset(dataset, rand_indices[:int(len(dataset) * 0.6)])
        dataset_eval = Subset(dataset, rand_indices[int(len(dataset) * 0.6):])

        ev = aso_pred_evaluator.RNABertEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_eval)

    if args.method == 'RNAFM':
        args.max_seq_len = MAX_SEQ_LEN["RNAFM"]
        args.replace_T = True
        args.replace_U = False
        tokenizer = RNATokenizer(args.vocab_path)
        dataset = aso_pred_dataset.ASOTokenDataset(
            fpath=args.dataset)
        rand_indices = torch.randperm(len(dataset)).tolist()

        dataset_train = Subset(dataset, rand_indices[:int(len(dataset) * 0.6)])
        dataset_eval = Subset(dataset, rand_indices[int(len(dataset) * 0.6):])

        ev = aso_pred_evaluator.RNAFMEvaluator(args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_eval)