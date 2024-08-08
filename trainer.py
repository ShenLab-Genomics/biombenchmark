import argparse
from evaluator import splice_evaluator
from dataset import splice_dataset
from transformers import AutoTokenizer


def train_splice_SpliceBERT(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ev = splice_evaluator.SpliceBERTEvaluator(train=True, tissue_num=15,
                                              model_path=args.model_path,
                                              tokenizer=tokenizer)
    train_dataset = splice_dataset.SpliceBERTDataset(
        'dataset/splice_data/gtex_500_15tis/dataset_train_debug.h5',
        tokenizer=tokenizer,
        dnabert_k=None,
        max_len=500)
    test_dataset = splice_dataset.SpliceBERTDataset(
        'dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5',
        tokenizer=tokenizer,
        dnabert_k=None,
        max_len=500)
    ev.train(args, train_data=train_dataset, test_data=test_dataset)


def train_splice_DNABERT(args):
    print('DNABERT')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path)
    print('done')
    ev = splice_evaluator.DNABERTEvaluator(
        train=True, tissue_num=15,
        model_path=args.model_path,
        tokenizer=tokenizer
    )
    train_dataset = splice_dataset.SpliceBERTDataset(
        'dataset/splice_data/gtex_500_15tis/dataset_train_debug.h5',
        tokenizer=tokenizer,
        dnabert_k=3,
        max_len=500)
    test_dataset = splice_dataset.SpliceBERTDataset(
        'dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5',
        tokenizer=tokenizer,
        dnabert_k=3,
        max_len=500)
    ev.train(args, train_data=train_dataset, test_data=test_dataset)
    pass

def train_splice_DNABERT2(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        "--method", choices=['SpTransformer', 'DNABERT1','DNABERT2', 'SpliceBERT'])
    parser.add_argument("--task", choices=['rna', 'splicing', 'm6a', 'rbp'],
                        default='splicing')
    parser.add_argument("--resume", default=False)
    parser.add_argument("--debug", default=False)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--output_dir", default='model/fine_tuned/default')
    parser.add_argument("--patience", default=5)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    args = parser.parse_args()

    if args.task == 'splicing':
        print(args.method)
        if args.method == 'DNABERT1':
            train_splice_DNABERT(args)
        if args.method == 'SpliceBERT':
            train_splice_SpliceBERT(args)
