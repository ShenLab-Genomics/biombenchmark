import argparse
from transformers import AutoTokenizer
from evaluator import splice_evaluator
from transformers import AutoTokenizer
from dataset import splice_dataset

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
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=1)
    parser.add_argument("--dataset_train", type=str)
    parser.add_argument("--dataset_test", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="topk,prauc,rocauc",)

    args = parser.parse_args()

    if args.method == 'SpliceBERT':
        args.max_seq_len = MAX_SEQ_LEN["SpliceBERT"]
        args.replace_T = False
        args.replace_U = True
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path)

        dataset_train = splice_dataset.SpliceDataset(fpath=args.dataset_train)
        dataset_test = splice_dataset.SpliceDataset(fpath=args.dataset_test)

        ev = splice_evaluator.SpliceBERTEvaluatorSeqCls(
            args, tokenizer=tokenizer)
        ev.run(args, dataset_train, dataset_test)
